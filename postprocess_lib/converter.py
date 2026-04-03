"""
Parallel MPI version of the standalone tools/convert_txt_to_hdf5.py script

TXT -> H5: Fully parallel — rank 0 builds a byte-offset index in one scan,
           then all ranks seek+read their own chunks and write to HDF5 in parallel.
H5 -> TXT: Serial on rank 0 — parallel text writing requires pre-computing
           per-row byte offsets (non-trivial), so not worth the complexity.

NOTE: driver='mpio' does not support gzip compression.  Files will be larger
      than the serial gzip'd HDF5 output, but still smaller than raw text
      (binary float64 vs ASCII %.16e).
"""

from __future__ import annotations

import io
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from mpi4py import rc

rc.initialize = False
rc.finalize = False

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CHUNK_SIZE = 1_000_000


def split_axis(length, parts):
    """Split a 1D index range into nearly equal contiguous chunks."""
    base, remainder = divmod(length, parts)
    start = 0
    chunks = []
    for i in range(parts):
        stop = start + base + (1 if i < remainder else 0)
        chunks.append((start, stop))
        start = stop
    return chunks


def is_structured_velocity_hdf5(h5_file):
    """Return True for the FFT-ready HDF5 schema."""
    return (
        "grid" in h5_file
        and "fields" in h5_file
        and "x" in h5_file["grid"]
        and "y" in h5_file["grid"]
        and "z" in h5_file["grid"]
        and "vx" in h5_file["fields"]
        and "vy" in h5_file["fields"]
        and "vz" in h5_file["fields"]
    )


def parse_header_metadata(header_lines):
    """Extract step/time metadata from the preserved text header."""
    step_number = "unknown"
    time_value = 0.0

    for line in header_lines:
        if "Cycle" in line or "Step" in line:
            match = re.search(r"(?:Cycle|Step)\s*[:=]\s*(\d+)", line)
            if match:
                step_number = match.group(1)
        if "Time" in line:
            match = re.search(r"Time\s*[:=]\s*([0-9.eE+-]+)", line)
            if match:
                time_value = float(match.group(1))

    return step_number, time_value


def merge_local_uniques(local_values):
    """Gather rank-local coordinate sets and merge them on rank 0."""
    gathered = comm.gather(np.asarray(local_values, dtype=np.float64), root=0)
    merged = None
    if rank == 0:
        nonempty = [arr for arr in gathered if arr.size > 0]
        merged = np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, dtype=np.float64)
    return comm.bcast(merged, root=0)


def validate_uniform_axis(axis, name):
    """Return the axis spacing and raise when the grid is not uniform."""
    if len(axis) <= 1:
        return 1.0
    diffs = np.diff(axis)
    spacing = float(diffs[0])
    tol = max(1.0, abs(spacing)) * 1.0e-8
    if not np.allclose(diffs, spacing, rtol=0.0, atol=tol):
        raise ValueError(f"Axis '{name}' is not uniformly spaced.")
    return spacing


def compute_axis_indices(values, axis, spacing, name):
    """Map coordinates onto integer grid indices using rounded spacing-based indexing."""
    if len(axis) == 1:
        return np.zeros(values.shape, dtype=np.int64)
    idx = np.rint((values - axis[0]) / spacing).astype(np.int64)
    if np.any(idx < 0) or np.any(idx >= len(axis)):
        raise ValueError(f"Axis index out of bounds while mapping '{name}' coordinates.")
    tol = max(1.0, abs(spacing)) * 1.0e-8
    if not np.allclose(axis[idx], values, rtol=0.0, atol=tol):
        raise ValueError(f"Axis '{name}' contains coordinates that do not map cleanly to the structured grid.")
    return idx


def open_h5_for_read(path):
    """Use MPI-IO when available; otherwise fall back to independent HDF5 reads."""
    if h5py.get_config().mpi:
        return h5py.File(path, "r", driver="mpio", comm=comm)
    return h5py.File(path, "r")


def get_txt_header(txt_file_path):
    """Rank 0 only: find header length. Returns (header_lines, count)."""
    print("  Analyzing text file header...")
    header_lines, header_count = [], 0
    try:
        with open(txt_file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    header_lines.append(line)
                    header_count += 1
                    continue
                try:
                    [float(x) for x in stripped.split()]
                    break
                except ValueError:
                    header_lines.append(line)
                    header_count += 1
        print(f"  Detected header length: {header_count} lines")
        return header_lines, header_count
    except Exception as exc:
        print(f"  Error analyzing file: {exc}")
        return None, 0


def build_chunk_index(txt_path, skip_count):
    """Rank 0 only. Scan file after the header and record (byte_offset, num_rows) per block."""
    chunks = []
    with open(txt_path, "rb") as handle:
        for _ in range(skip_count):
            handle.readline()
        while True:
            offset = handle.tell()
            count = 0
            for _ in range(CHUNK_SIZE):
                if not handle.readline():
                    break
                count += 1
            if count == 0:
                break
            chunks.append((offset, count))
    return chunks


def read_chunk_at_offset(txt_path, byte_offset, num_rows):
    """Seek directly to byte_offset, read exactly num_rows lines, and parse the chunk."""
    with open(txt_path, "rb") as handle:
        handle.seek(byte_offset)
        buf = b"".join(handle.readline() for _ in range(num_rows))
    return pd.read_csv(io.BytesIO(buf), header=None, sep=r"\s+").values.astype(np.float64)


def discover_grid_from_chunks(txt_path, chunk_index):
    """Parallel pass over the text file to discover the structured grid."""
    local_x = np.empty(0, dtype=np.float64)
    local_y = np.empty(0, dtype=np.float64)
    local_z = np.empty(0, dtype=np.float64)

    for ci in range(rank, len(chunk_index), size):
        byte_off, nrows = chunk_index[ci]
        chunk_data = read_chunk_at_offset(txt_path, byte_off, nrows)

        x_vals = np.unique(np.round(chunk_data[:, 0], 10))
        y_vals = np.unique(np.round(chunk_data[:, 1], 10))
        z_vals = np.unique(np.round(chunk_data[:, 2], 10))

        local_x = np.unique(np.concatenate((local_x, x_vals)))
        local_y = np.unique(np.concatenate((local_y, y_vals)))
        local_z = np.unique(np.concatenate((local_z, z_vals)))

    x_unique = merge_local_uniques(local_x)
    y_unique = merge_local_uniques(local_y)
    z_unique = merge_local_uniques(local_z)

    dx = validate_uniform_axis(x_unique, "x")
    dy = validate_uniform_axis(y_unique, "y")
    dz = validate_uniform_axis(z_unique, "z")
    return x_unique, y_unique, z_unique, dx, dy, dz


def redistribute_to_xslabs(txt_path, chunk_index, x_unique, y_unique, z_unique, dx, dy, dz):
    """Read rows in parallel, map them to grid indices, and redistribute by x-slab."""
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    x_ranges = split_axis(nx, size)
    x_stops = np.array([stop for _, stop in x_ranges], dtype=np.int64)

    send_bins = [[] for _ in range(size)]
    running_sum_sq = 0.0

    for ci in range(rank, len(chunk_index), size):
        byte_off, nrows = chunk_index[ci]
        chunk_data = read_chunk_at_offset(txt_path, byte_off, nrows)

        x_vals = np.round(chunk_data[:, 0], 10)
        y_vals = np.round(chunk_data[:, 1], 10)
        z_vals = np.round(chunk_data[:, 2], 10)
        vx = chunk_data[:, 3]
        vy = chunk_data[:, 4]
        vz = chunk_data[:, 5]

        ix = compute_axis_indices(x_vals, x_unique, dx, "x")
        iy = compute_axis_indices(y_vals, y_unique, dy, "y")
        iz = compute_axis_indices(z_vals, z_unique, dz, "z")

        dest_ranks = np.searchsorted(x_stops, ix, side="right")
        for dest_rank in np.unique(dest_ranks):
            mask = dest_ranks == dest_rank
            packed = np.column_stack(
                (
                    ix[mask].astype(np.float64),
                    iy[mask].astype(np.float64),
                    iz[mask].astype(np.float64),
                    vx[mask],
                    vy[mask],
                    vz[mask],
                )
            )
            send_bins[int(dest_rank)].append(packed)

        running_sum_sq += np.sum(vx**2 + vy**2 + vz**2, dtype=np.float64)

    send_arrays = []
    for bins in send_bins:
        if bins:
            send_arrays.append(np.ascontiguousarray(np.vstack(bins), dtype=np.float64))
        else:
            send_arrays.append(np.empty((0, 6), dtype=np.float64))

    send_counts = np.array([arr.size for arr in send_arrays], dtype=np.int32)
    recv_counts = np.empty(size, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)

    send_displs = np.zeros(size, dtype=np.int32)
    recv_displs = np.zeros(size, dtype=np.int32)
    if size > 1:
        send_displs[1:] = np.cumsum(send_counts[:-1], dtype=np.int32)
        recv_displs[1:] = np.cumsum(recv_counts[:-1], dtype=np.int32)

    sendbuf = (
        np.concatenate([arr.ravel(order="C") for arr in send_arrays])
        if np.any(send_counts)
        else np.empty(0, dtype=np.float64)
    )
    recvbuf = np.empty(int(np.sum(recv_counts)), dtype=np.float64)

    comm.Alltoallv(
        [sendbuf, send_counts, send_displs, MPI.DOUBLE],
        [recvbuf, recv_counts, recv_displs, MPI.DOUBLE],
    )

    x_start, x_stop = x_ranges[rank]
    local_nx = x_stop - x_start
    local_vx = np.zeros((local_nx, ny, nz), dtype=np.float64)
    local_vy = np.zeros((local_nx, ny, nz), dtype=np.float64)
    local_vz = np.zeros((local_nx, ny, nz), dtype=np.float64)
    filled = np.zeros((local_nx, ny, nz), dtype=bool)

    if recvbuf.size:
        received = recvbuf.reshape((-1, 6), order="C")
        li = received[:, 0].astype(np.int64) - x_start
        lj = received[:, 1].astype(np.int64)
        lk = received[:, 2].astype(np.int64)

        local_vx[li, lj, lk] = received[:, 3]
        local_vy[li, lj, lk] = received[:, 4]
        local_vz[li, lj, lk] = received[:, 5]
        filled[li, lj, lk] = True

    if filled.size and np.count_nonzero(filled) != filled.size:
        raise ValueError(f"Rank {rank} did not receive a complete x-slab during redistribution.")

    return (x_start, x_stop), local_vx, local_vy, local_vz, running_sum_sq


def write_structured_h5(
    h5_path,
    x_unique,
    y_unique,
    z_unique,
    local_x_bounds,
    local_vx,
    local_vy,
    local_vz,
    header_lines,
):
    """Write the structured FFT-ready HDF5 file, using MPI-IO when available."""
    global_shape = (len(x_unique), len(y_unique), len(z_unique))
    x_start, x_stop = local_x_bounds

    if rank == 0 and os.path.exists(h5_path):
        os.remove(h5_path)
    comm.Barrier()

    if h5py.get_config().mpi:
        with h5py.File(h5_path, "w", driver="mpio", comm=comm) as hf:
            fields = hf.create_group("fields")
            vx_dset = fields.create_dataset("vx", shape=global_shape, dtype="float64")
            vy_dset = fields.create_dataset("vy", shape=global_shape, dtype="float64")
            vz_dset = fields.create_dataset("vz", shape=global_shape, dtype="float64")

            if x_stop > x_start:
                slab = slice(x_start, x_stop)
                vx_dset[slab, :, :] = local_vx
                vy_dset[slab, :, :] = local_vy
                vz_dset[slab, :, :] = local_vz
    else:
        gathered = comm.gather((x_start, x_stop, local_vx, local_vy, local_vz), root=0)
        if rank == 0:
            with h5py.File(h5_path, "w") as hf:
                fields = hf.create_group("fields")
                vx_dset = fields.create_dataset("vx", shape=global_shape, dtype="float64")
                vy_dset = fields.create_dataset("vy", shape=global_shape, dtype="float64")
                vz_dset = fields.create_dataset("vz", shape=global_shape, dtype="float64")
                for slab_start, slab_stop, slab_vx, slab_vy, slab_vz in gathered:
                    if slab_stop > slab_start:
                        slab = slice(slab_start, slab_stop)
                        vx_dset[slab, :, :] = slab_vx
                        vy_dset[slab, :, :] = slab_vy
                        vz_dset[slab, :, :] = slab_vz

    comm.Barrier()

    if rank == 0:
        step_number, time_value = parse_header_metadata(header_lines)
        with h5py.File(h5_path, "a") as hf:
            grid = hf.create_group("grid")
            grid.create_dataset("x", data=x_unique)
            grid.create_dataset("y", data=y_unique)
            grid.create_dataset("z", data=z_unique)

            dt = h5py.string_dtype(encoding="utf-8")
            hf.create_dataset(
                "header",
                data=np.array([str(line) for line in header_lines], dtype=object),
                dtype=dt,
            )

            hf.attrs["schema"] = "structured_velocity_v1"
            hf.attrs["periodic_duplicate_last"] = True
            hf.attrs["Nx"] = int(global_shape[0])
            hf.attrs["Ny"] = int(global_shape[1])
            hf.attrs["Nz"] = int(global_shape[2])
            hf.attrs["fft_nx"] = int(max(global_shape[0] - 1, 1))
            hf.attrs["fft_ny"] = int(max(global_shape[1] - 1, 1))
            hf.attrs["fft_nz"] = int(max(global_shape[2] - 1, 1))
            hf.attrs["dx"] = float(x_unique[1] - x_unique[0]) if len(x_unique) > 1 else 1.0
            hf.attrs["dy"] = float(y_unique[1] - y_unique[0]) if len(y_unique) > 1 else 1.0
            hf.attrs["dz"] = float(z_unique[1] - z_unique[0]) if len(z_unique) > 1 else 1.0
            hf.attrs["xmin"] = float(x_unique[0])
            hf.attrs["xmax"] = float(x_unique[-1])
            hf.attrs["ymin"] = float(y_unique[0])
            hf.attrs["ymax"] = float(y_unique[-1])
            hf.attrs["zmin"] = float(z_unique[0])
            hf.attrs["zmax"] = float(z_unique[-1])
            hf.attrs["step"] = step_number
            hf.attrs["time"] = float(time_value)


def convert_txt_to_h5_parallel(txt_path, h5_path):
    """Convert one TXT file to structured HDF5 in parallel."""
    header_lines = chunk_index = None
    total_rows = 0

    if rank == 0:
        header_lines, skip_count = get_txt_header(txt_path)
        if header_lines is not None:
            print("  Building chunk index (rank 0 scans file once)...")
            chunk_index = build_chunk_index(txt_path, skip_count)
            total_rows = sum(n for _, n in chunk_index)
            print(f"  {total_rows} rows | {len(chunk_index)} chunks | {size} MPI ranks")

    header_lines = comm.bcast(header_lines, root=0)
    chunk_index = comm.bcast(chunk_index, root=0)
    total_rows = comm.bcast(total_rows, root=0)

    if header_lines is None or not chunk_index:
        return None, 0

    if rank == 0:
        print("  Discovering structured grid in parallel...")

    x_unique, y_unique, z_unique, dx, dy, dz = discover_grid_from_chunks(txt_path, chunk_index)
    expected_rows = len(x_unique) * len(y_unique) * len(z_unique)

    if rank == 0:
        print(f"  Structured grid: {len(x_unique)} x {len(y_unique)} x {len(z_unique)}")
        print(f"  Grid spacing: dx={dx:.16e}, dy={dy:.16e}, dz={dz:.16e}")
        if expected_rows != total_rows:
            raise ValueError(
                f"Structured grid size ({expected_rows}) does not match text row count ({total_rows})."
            )
        print("  Redistributing rows into FFT-ready x-slabs...")

    local_x_bounds, local_vx, local_vy, local_vz, running_sum_sq = redistribute_to_xslabs(
        txt_path, chunk_index, x_unique, y_unique, z_unique, dx, dy, dz
    )

    if rank == 0:
        if h5py.get_config().mpi:
            print("  Writing structured HDF5 in parallel...")
        else:
            print("  h5py MPI support is unavailable; falling back to serial HDF5 assembly on rank 0...")

    write_structured_h5(
        h5_path,
        x_unique,
        y_unique,
        z_unique,
        local_x_bounds,
        local_vx,
        local_vy,
        local_vz,
        header_lines,
    )

    global_sum_sq = comm.reduce(running_sum_sq, op=MPI.SUM, root=0)
    tke = None
    if rank == 0:
        tke = 0.5 * (global_sum_sq / total_rows)
        print(f"  Conversion complete. Total rows: {total_rows}")
    tke = comm.bcast(tke, root=0)

    return tke, total_rows


def convert_h5_to_txt_chunked(h5_path, txt_path):
    """Convert one HDF5 file back to text on rank 0."""
    if rank != 0:
        return None, 0

    try:
        print("  Converting H5 -> TXT (serial, rank 0)...")
        with h5py.File(h5_path, "r") as hf:
            print("  HDF5 read mode: serial h5py read on rank 0")
            header_lines = []
            if "header" in hf:
                for line in hf["header"][:]:
                    restored = line.decode("utf-8") if isinstance(line, bytes) else str(line)
                    header_lines.append(restored)
            header_lines.insert(0, "# Data restored from HDF5 conversion (Original source: Text file)\n")

            running_sum_sq = 0.0
            processed_rows = 0

            with open(txt_path, "w", encoding="utf-8") as txt_file:
                for line in header_lines:
                    txt_file.write(line)
                if "data" in hf:
                    dset = hf["data"]
                    total_rows_h5 = dset.shape[0]
                    for i in range(0, total_rows_h5, CHUNK_SIZE):
                        chunk_data = dset[i : i + CHUNK_SIZE]
                        np.savetxt(txt_file, chunk_data, fmt="%.16e", delimiter=" ")
                        vx, vy, vz = chunk_data[:, 3], chunk_data[:, 4], chunk_data[:, 5]
                        running_sum_sq += np.sum(vx**2 + vy**2 + vz**2)
                        processed_rows += len(chunk_data)
                        print(f"    {processed_rows}/{total_rows_h5} rows...", end="\r")
                elif is_structured_velocity_hdf5(hf):
                    x = np.asarray(hf["grid"]["x"][:], dtype=np.float64)
                    y = np.asarray(hf["grid"]["y"][:], dtype=np.float64)
                    z = np.asarray(hf["grid"]["z"][:], dtype=np.float64)
                    vx_dset = hf["fields"]["vx"]
                    vy_dset = hf["fields"]["vy"]
                    vz_dset = hf["fields"]["vz"]

                    total_rows_h5 = vx_dset.shape[0] * vx_dset.shape[1] * vx_dset.shape[2]
                    x_chunk = max(1, CHUNK_SIZE // max(len(y) * len(z), 1))

                    for i in range(0, len(x), x_chunk):
                        i_end = min(i + x_chunk, len(x))
                        x_block = x[i:i_end]
                        vx_block = np.asarray(vx_dset[i:i_end, :, :], dtype=np.float64)
                        vy_block = np.asarray(vy_dset[i:i_end, :, :], dtype=np.float64)
                        vz_block = np.asarray(vz_dset[i:i_end, :, :], dtype=np.float64)

                        X, Y, Z = np.meshgrid(x_block, y, z, indexing="ij")
                        chunk_data = np.column_stack(
                            (
                                X.ravel(order="C"),
                                Y.ravel(order="C"),
                                Z.ravel(order="C"),
                                vx_block.ravel(order="C"),
                                vy_block.ravel(order="C"),
                                vz_block.ravel(order="C"),
                            )
                        )

                        np.savetxt(txt_file, chunk_data, fmt="%.16e", delimiter=" ")
                        running_sum_sq += np.sum(vx_block**2 + vy_block**2 + vz_block**2, dtype=np.float64)
                        processed_rows += len(chunk_data)
                        print(f"    {processed_rows}/{total_rows_h5} rows...", end="\r")
                else:
                    print("  Error: No recognized velocity dataset in HDF5 file.")
                    return None, 0

            print("\n  Conversion complete.")
            return 0.5 * (running_sum_sq / processed_rows), processed_rows

    except Exception as exc:
        print(f"\n  Error during H5->TXT: {exc}")
        return None, 0


def calculate_file_tke_parallel(file_path):
    """Return the same (tke, total_rows) on all ranks."""
    if rank == 0:
        print(f"  Verifying: {os.path.basename(file_path)}...")

    _, ext = os.path.splitext(file_path)
    running_sum_sq = 0.0
    local_rows = 0

    try:
        if ext == ".h5":
            if rank == 0:
                if h5py.get_config().mpi:
                    print("  HDF5 verification read mode: MPI-enabled parallel HDF5")
                else:
                    print("  HDF5 verification read mode: serial h5py on each rank (independent reads)")
            with open_h5_for_read(file_path) as hf:
                if "data" in hf:
                    dset = hf["data"]
                    total = dset.shape[0]
                    for start in range(rank * CHUNK_SIZE, total, size * CHUNK_SIZE):
                        chunk = dset[start : start + CHUNK_SIZE]
                        vx, vy, vz = chunk[:, 3], chunk[:, 4], chunk[:, 5]
                        running_sum_sq += np.sum(vx**2 + vy**2 + vz**2, dtype=np.float64)
                        local_rows += len(chunk)
                elif is_structured_velocity_hdf5(hf):
                    vx_dset = hf["fields"]["vx"]
                    vy_dset = hf["fields"]["vy"]
                    vz_dset = hf["fields"]["vz"]
                    x_ranges = split_axis(vx_dset.shape[0], size)
                    x_start, x_stop = x_ranges[rank]
                    if x_stop > x_start:
                        vx = np.asarray(vx_dset[x_start:x_stop, :, :], dtype=np.float64)
                        vy = np.asarray(vy_dset[x_start:x_stop, :, :], dtype=np.float64)
                        vz = np.asarray(vz_dset[x_start:x_stop, :, :], dtype=np.float64)
                        running_sum_sq += np.sum(vx**2 + vy**2 + vz**2, dtype=np.float64)
                        local_rows += vx.size
                else:
                    raise ValueError("No recognized velocity dataset in HDF5 file.")
        else:
            if rank == 0:
                _, skip = get_txt_header(file_path)
                reader = pd.read_csv(file_path, skiprows=skip, header=None, sep=r"\s+", chunksize=CHUNK_SIZE)
                for chunk_df in reader:
                    chunk = chunk_df.values.astype(np.float64)
                    vx, vy, vz = chunk[:, 3], chunk[:, 4], chunk[:, 5]
                    running_sum_sq += np.sum(vx**2 + vy**2 + vz**2)
                    local_rows += len(chunk)

        global_sum_sq = comm.reduce(running_sum_sq, op=MPI.SUM, root=0)
        total_rows = comm.reduce(local_rows, op=MPI.SUM, root=0)

        tke = None
        if rank == 0:
            tke = 0.5 * (global_sum_sq / total_rows) if total_rows > 0 else 0.0
        tke = comm.bcast(tke, root=0)
        total_rows = comm.bcast(total_rows, root=0)

        return tke, total_rows

    except Exception as exc:
        if rank == 0:
            print(f"  Error verifying file: {exc}")
        return None, 0


def print_storage_stats(path1, path2):
    """Print input/output size comparison."""
    size1 = os.path.getsize(path1)
    size2 = os.path.getsize(path2)
    diff = size1 - size2
    print("-" * 40)
    print("Storage Comparison:")
    print(f"  Input:  {size1 / (1024 * 1024):.2f} MB")
    print(f"  Output: {size2 / (1024 * 1024):.2f} MB")
    if diff > 0:
        print(f"  Saved:  {diff / (1024 * 1024):.2f} MB ({(diff / size1) * 100:.2f}%)")
    else:
        print(f"  Growth: {abs(diff) / (1024 * 1024):.2f} MB")
    print("-" * 40)


def convert_file(input_path):
    """Convert one TXT/HDF5 file and verify the result."""
    if rank == 0:
        print(f"\nProcessing: {input_path}")

    exists = comm.bcast(os.path.exists(input_path) if rank == 0 else False, root=0)
    if not exists:
        if rank == 0:
            print("Error: File not found.")
        return False

    base, ext = os.path.splitext(input_path)
    input_tke = None
    input_rows = 0

    if ext == ".txt":
        output_path = base + ".h5"
        input_tke, input_rows = convert_txt_to_h5_parallel(input_path, output_path)
    elif ext == ".h5":
        output_path = base + ".txt"
        input_tke, input_rows = convert_h5_to_txt_chunked(input_path, output_path)
        input_tke = comm.bcast(input_tke, root=0)
        input_rows = comm.bcast(input_rows, root=0)
    else:
        if rank == 0:
            print(f"Error: Unsupported extension '{ext}'.")
        return False

    if input_tke is None:
        return False

    output_tke, output_rows = calculate_file_tke_parallel(output_path)
    if output_tke is None:
        return False

    success = False
    if rank == 0:
        print(f"  Original TKE:       {input_tke:.16f}")
        print(f"  Reconstructed TKE:  {output_tke:.16f}")
        print(f"  Original Rows:      {input_rows}")
        print(f"  Reconstructed Rows: {output_rows}")

        tke_match = np.isclose(input_tke, output_tke, atol=1e-12)
        row_match = input_rows == output_rows

        if tke_match and row_match:
            print("  SUCCESS: TKE and Row Counts match.")
            print_storage_stats(input_path, output_path)
            if ext == ".txt":
                print(f"  Deleting original: {input_path}")
                try:
                    os.remove(input_path)
                except OSError as exc:
                    print(f"  Warning: Could not delete: {exc}")
            success = True
        else:
            print("\n" + "!" * 60)
            print("  !!! ERROR: DATA INTEGRITY FAILED !!!")
            if not tke_match:
                print(f"  TKE mismatch! Diff: {abs(input_tke - output_tke):.2e}")
            if not row_match:
                print(f"  Row mismatch! Input: {input_rows}, Output: {output_rows}")
            print("!" * 60 + "\n")

    return comm.bcast(success, root=0)


def main(argv=None):
    """CLI entrypoint for converter compatibility wrapper."""
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        if rank == 0:
            print("\nUsage: mpirun -n N python tools/convert_txt_to_hdf5.py <file1> [file2 ...]")
        return 0

    files = argv
    failures = 0

    if rank == 0:
        print(f"Batch processing {len(files)} file(s) with {size} MPI ranks...")

    for file_path in files:
        try:
            if not convert_file(file_path):
                failures += 1
        except Exception as exc:
            if rank == 0:
                print(f"  CRITICAL ERROR processing {file_path}: {exc}")
            failures += 1

    if rank == 0:
        print("\n" + "=" * 40)
        if failures == 0:
            print(f"Batch completed successfully. {len(files)} file(s) processed.")
        else:
            print(f"Batch completed with ERRORS. {failures}/{len(files)} file(s) failed.")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
