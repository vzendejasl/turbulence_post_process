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
DEFAULT_DEDALUS_IMPORT_X_BLOCK_SIZE = 1
DEDALUS_IMPORT_X_BLOCK_SIZE_ENV = "TPP_DEDALUS_IMPORT_X_BLOCK_SIZE"
UINT64_MAX = (1 << 64) - 1


def resolve_dedalus_import_x_block_size(x_block_size=None):
    """Return the number of x-planes to stream per Dedalus import read."""
    if x_block_size is None:
        value = os.environ.get(DEDALUS_IMPORT_X_BLOCK_SIZE_ENV, DEFAULT_DEDALUS_IMPORT_X_BLOCK_SIZE)
    else:
        value = x_block_size

    try:
        block_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid Dedalus import x-block size {value!r}. "
            "Expected a positive integer."
        ) from exc

    if block_size < 1:
        raise ValueError(
            f"Invalid Dedalus import x-block size {block_size}. "
            "Expected a positive integer."
        )
    return block_size


def regular_hyperslab_geometry(space):
    """Return integer start and span for a regular HDF5 hyperslab selection."""
    start, stride, count, block = space.get_regular_hyperslab()
    start = tuple(int(value) for value in start)
    stride = tuple(int(value) for value in stride)
    count = tuple(int(value) for value in count)
    block = tuple(int(value) for value in block)

    span = []
    for stride_i, count_i, block_i in zip(stride, count, block):
        if count_i == UINT64_MAX:
            span.append(block_i)
        else:
            span.append((count_i - 1) * stride_i + block_i)
    return start, tuple(span)


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


def is_structured_scalar_hdf5(h5_file):
    """Return True for the standalone structured scalar HDF5 schema."""
    if "grid" not in h5_file or "fields" not in h5_file:
        return False
    if "x" not in h5_file["grid"] or "y" not in h5_file["grid"] or "z" not in h5_file["grid"]:
        return False
    field_names = [name for name, value in h5_file["fields"].items() if isinstance(value, h5py.Dataset)]
    if len(field_names) != 1:
        return False
    field_name = field_names[0]
    return field_name not in {"vx", "vy", "vz"}


def is_dedalus_field_output_hdf5(h5_file):
    """Return True for Dedalus field-output files with vector velocity tasks."""
    if "tasks" not in h5_file or "scales" not in h5_file:
        return False
    if "u" not in h5_file["tasks"]:
        return False
    u_task = h5_file["tasks"]["u"]
    return u_task.ndim == 5 and u_task.shape[1] == 3


def dedalus_coordinate_dataset_names(h5_file):
    """Return the Dedalus coordinate scale dataset names for x/y/z."""
    scale_keys = list(h5_file["scales"].keys())
    names = {}
    for axis in ("x", "y", "z"):
        matches = [key for key in scale_keys if key.startswith(f"{axis}_hash_")]
        if len(matches) != 1:
            raise ValueError(
                f"Could not identify the Dedalus {axis}-coordinate dataset uniquely. "
                f"Found: {matches!r}"
            )
        names[axis] = matches[0]
    return names["x"], names["y"], names["z"]


def dedalus_snapshot_info(path, write_index=-1):
    """Read one Dedalus field-output file and describe the selected snapshot."""
    with h5py.File(path, "r") as hf:
        if not is_dedalus_field_output_hdf5(hf):
            raise ValueError(f"{path} is not a Dedalus field-output HDF5 file.")

        x_name, y_name, z_name = dedalus_coordinate_dataset_names(hf)
        x_coords = np.asarray(hf["scales"][x_name][:], dtype=np.float64)
        y_coords = np.asarray(hf["scales"][y_name][:], dtype=np.float64)
        z_coords = np.asarray(hf["scales"][z_name][:], dtype=np.float64)

        u_task = hf["tasks"]["u"]
        num_writes = int(u_task.shape[0])
        selected = int(write_index)
        if selected < 0:
            selected += num_writes
        if selected < 0 or selected >= num_writes:
            raise IndexError(
                f"Requested Dedalus write index {write_index} is out of bounds for {num_writes} writes."
            )

        sim_time = float(np.asarray(hf["scales"]["sim_time"][selected]))
        write_number = int(np.asarray(hf["scales"]["write_number"][selected]))
        cycle = None
        if "cycle" in hf["tasks"]:
            cycle = int(np.asarray(hf["tasks"]["cycle"][selected]).reshape(-1)[0])

        return {
            "shape": (len(x_coords), len(y_coords), len(z_coords)),
            "x": x_coords,
            "y": y_coords,
            "z": z_coords,
            "selected_index": selected,
            "num_writes": num_writes,
            "sim_time": sim_time,
            "write_number": write_number,
            "cycle": cycle,
            "set_number": int(hf.attrs.get("set_number", 0)),
            "handler_name": str(hf.attrs.get("handler_name", "dedalus_fields")),
        }


def dedalus_snapshot_output_path(input_path, snapshot_info):
    """Return the structured-output path for one imported Dedalus snapshot.

    Output is placed under ``post_process/cycle_<step>/`` next to the input
    file, where *step* is the cycle number (or write number when the cycle is
    unavailable).
    """
    input_dir = os.path.dirname(os.path.abspath(input_path))
    step = snapshot_info["cycle"] if snapshot_info["cycle"] is not None else snapshot_info["write_number"]
    cycle_dir = os.path.join(input_dir, "post_process", f"cycle_{step}")
    os.makedirs(cycle_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(cycle_dir, f"{base}_write{snapshot_info['write_number']:05d}.h5")


def dedalus_header_lines(input_path, snapshot_info):
    """Create header-like metadata lines for an imported Dedalus snapshot."""
    cycle = snapshot_info["cycle"]
    cycle_text = "unknown" if cycle is None else str(cycle)
    return [
        f"# Imported from Dedalus field output: {os.path.abspath(input_path)}\n",
        f"# Write Number: {snapshot_info['write_number']}\n",
        f"# Cycle: {cycle_text}\n",
        f"# Time: {snapshot_info['sim_time']:.16e}\n",
    ]


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


def sanitize_field_name(name):
    """Return a dataset-safe field name derived from a sampled-data label."""
    field_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip().lower())
    field_name = re.sub(r"_+", "_", field_name).strip("_")
    if not field_name:
        raise ValueError("Unable to derive a valid field name from the sampled-data header.")
    if field_name[0].isdigit():
        field_name = f"field_{field_name}"
    return field_name


def parse_sampled_data_field_name(header_lines, fallback_name):
    """Extract the sampled quantity name from the preserved text header."""
    for line in header_lines:
        match = re.search(r"Sampled Data\s*,\s*(.+?)\s*$", line)
        if match:
            display_name = match.group(1).strip()
            return sanitize_field_name(display_name), display_name

    display_name = str(fallback_name).strip()
    return sanitize_field_name(display_name), display_name


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


def open_h5_for_independent_read(path):
    """Open one HDF5 file without MPI-IO, even inside an MPI run."""
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


def read_structured_grid(path):
    """Read the structured HDF5 grid coordinates on rank 0 and broadcast them."""
    x_unique = y_unique = z_unique = None
    if rank == 0:
        with h5py.File(path, "r") as hf:
            if not is_structured_velocity_hdf5(hf):
                raise ValueError(f"{path} is not a structured FFT-ready velocity HDF5 file.")
            x_unique = np.asarray(hf["grid"]["x"][:], dtype=np.float64)
            y_unique = np.asarray(hf["grid"]["y"][:], dtype=np.float64)
            z_unique = np.asarray(hf["grid"]["z"][:], dtype=np.float64)
    x_unique = comm.bcast(x_unique, root=0)
    y_unique = comm.bcast(y_unique, root=0)
    z_unique = comm.bcast(z_unique, root=0)
    return x_unique, y_unique, z_unique


def read_scalar_structured_h5_metadata(path):
    """Read one standalone structured scalar HDF5 file and broadcast its metadata."""
    field_name = display_name = None
    x_unique = y_unique = z_unique = None
    total_rows = 0

    if rank == 0:
        with h5py.File(path, "r") as hf:
            if not is_structured_scalar_hdf5(hf):
                raise ValueError(f"{path} is not a standalone structured scalar HDF5 file.")
            field_name = next(iter(hf["fields"].keys()))
            dataset = hf["fields"][field_name]
            display_name = dataset.attrs.get("display_name", hf.attrs.get("display_name", field_name))
            if isinstance(display_name, bytes):
                display_name = display_name.decode("utf-8")
            x_unique = np.asarray(hf["grid"]["x"][:], dtype=np.float64)
            y_unique = np.asarray(hf["grid"]["y"][:], dtype=np.float64)
            z_unique = np.asarray(hf["grid"]["z"][:], dtype=np.float64)
            total_rows = int(np.prod(dataset.shape))

    field_name = comm.bcast(field_name, root=0)
    display_name = comm.bcast(display_name, root=0)
    x_unique = comm.bcast(x_unique, root=0)
    y_unique = comm.bcast(y_unique, root=0)
    z_unique = comm.bcast(z_unique, root=0)
    total_rows = comm.bcast(total_rows, root=0)
    return field_name, display_name, x_unique, y_unique, z_unique, total_rows


def validate_matching_structured_grid(source_path, h5_path, source_grid, h5_grid):
    """Ensure an auxiliary structured scalar input matches the main structured HDF5 grid."""
    txt_x, txt_y, txt_z = source_grid
    h5_x, h5_y, h5_z = h5_grid

    for axis_name, txt_axis, h5_axis in (
        ("x", txt_x, h5_x),
        ("y", txt_y, h5_y),
        ("z", txt_z, h5_z),
    ):
        if len(txt_axis) != len(h5_axis) or not np.allclose(txt_axis, h5_axis, rtol=0.0, atol=1.0e-10):
            raise ValueError(
                f"Scalar field grid from {source_path} does not match {h5_path} along axis '{axis_name}'."
            )


def redistribute_scalar_to_xslabs(txt_path, chunk_index, x_unique, y_unique, z_unique, dx, dy, dz):
    """Read a scalar sampled-data TXT file in parallel and redistribute by x-slab."""
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    x_ranges = split_axis(nx, size)
    x_stops = np.array([stop for _, stop in x_ranges], dtype=np.int64)

    send_bins = [[] for _ in range(size)]
    local_rows = 0

    for ci in range(rank, len(chunk_index), size):
        byte_off, nrows = chunk_index[ci]
        chunk_data = read_chunk_at_offset(txt_path, byte_off, nrows)
        if chunk_data.shape[1] < 4:
            raise ValueError(f"Expected at least 4 columns in scalar sampled-data file {txt_path}.")

        x_vals = np.round(chunk_data[:, 0], 10)
        y_vals = np.round(chunk_data[:, 1], 10)
        z_vals = np.round(chunk_data[:, 2], 10)
        scalar_values = chunk_data[:, 3]

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
                    scalar_values[mask],
                )
            )
            send_bins[int(dest_rank)].append(packed)

        local_rows += len(chunk_data)

    send_arrays = []
    for bins in send_bins:
        if bins:
            send_arrays.append(np.ascontiguousarray(np.vstack(bins), dtype=np.float64))
        else:
            send_arrays.append(np.empty((0, 4), dtype=np.float64))

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
    local_scalar = np.zeros((local_nx, ny, nz), dtype=np.float64)
    filled = np.zeros((local_nx, ny, nz), dtype=bool)

    if recvbuf.size:
        received = recvbuf.reshape((-1, 4), order="C")
        li = received[:, 0].astype(np.int64) - x_start
        lj = received[:, 1].astype(np.int64)
        lk = received[:, 2].astype(np.int64)
        local_scalar[li, lj, lk] = received[:, 3]
        filled[li, lj, lk] = True

    if filled.size and np.count_nonzero(filled) != filled.size:
        raise ValueError(f"Rank {rank} did not receive a complete x-slab during scalar redistribution.")

    return (x_start, x_stop), local_scalar, local_rows


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


def write_structured_h5_metadata(
    h5_path,
    x_unique,
    y_unique,
    z_unique,
    header_lines,
    *,
    step_number=None,
    time_value=None,
    periodic_duplicate_last=True,
    extra_attrs=None,
):
    """Write shared grid/header metadata for the structured velocity schema."""
    global_shape = (len(x_unique), len(y_unique), len(z_unique))
    if step_number is None or time_value is None:
        parsed_step, parsed_time = parse_header_metadata(header_lines)
        if step_number is None:
            step_number = parsed_step
        if time_value is None:
            time_value = parsed_time

    with h5py.File(h5_path, "a") as hf:
        if "grid" in hf:
            del hf["grid"]
        if "header" in hf:
            del hf["header"]

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
        hf.attrs["periodic_duplicate_last"] = bool(periodic_duplicate_last)
        hf.attrs["Nx"] = int(global_shape[0])
        hf.attrs["Ny"] = int(global_shape[1])
        hf.attrs["Nz"] = int(global_shape[2])
        if periodic_duplicate_last:
            hf.attrs["fft_nx"] = int(max(global_shape[0] - 1, 1))
            hf.attrs["fft_ny"] = int(max(global_shape[1] - 1, 1))
            hf.attrs["fft_nz"] = int(max(global_shape[2] - 1, 1))
        else:
            hf.attrs["fft_nx"] = int(global_shape[0])
            hf.attrs["fft_ny"] = int(global_shape[1])
            hf.attrs["fft_nz"] = int(global_shape[2])
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
        if extra_attrs:
            for key, value in extra_attrs.items():
                hf.attrs[key] = value


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
    *,
    step_number=None,
    time_value=None,
    periodic_duplicate_last=True,
    extra_attrs=None,
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
        write_structured_h5_metadata(
            h5_path,
            x_unique,
            y_unique,
            z_unique,
            header_lines,
            step_number=step_number,
            time_value=time_value,
            periodic_duplicate_last=periodic_duplicate_last,
            extra_attrs=extra_attrs,
        )


def stream_dedalus_snapshot_to_structured_h5(
    input_path,
    output_path,
    snapshot_info,
    x_start,
    x_stop,
    x_block_size,
):
    """Stream one Dedalus velocity snapshot into the structured schema."""
    global_shape = snapshot_info["shape"]
    selected = snapshot_info["selected_index"]
    running_sum_sq = 0.0
    local_rows = 0

    if rank == 0 and os.path.exists(output_path):
        os.remove(output_path)
    comm.Barrier()

    with h5py.File(output_path, "w", driver="mpio", comm=comm) as out_hf:
        fields = out_hf.create_group("fields")
        output_dsets = (
            fields.create_dataset("vx", shape=global_shape, dtype="float64"),
            fields.create_dataset("vy", shape=global_shape, dtype="float64"),
            fields.create_dataset("vz", shape=global_shape, dtype="float64"),
        )

        with open_h5_for_independent_read(input_path) as in_hf:
            u_task = in_hf["tasks"]["u"]
            ny, nz = global_shape[1], global_shape[2]

            for block_start in range(x_start, x_stop, x_block_size):
                block_stop = min(block_start + x_block_size, x_stop)
                block_nx = block_stop - block_start
                if block_nx <= 0:
                    continue

                buffer = np.empty((1, 1, block_nx, ny, nz), dtype=np.float64)
                output_slab = np.s_[block_start:block_stop, :, :]
                for component, output_dset in enumerate(output_dsets):
                    u_task.read_direct(
                        buffer,
                        source_sel=np.s_[selected:selected + 1, component:component + 1, block_start:block_stop, :, :],
                    )
                    block_values = buffer[0, 0]
                    output_dset[output_slab] = block_values
                    running_sum_sq += np.sum(block_values**2, dtype=np.float64)

                local_rows += block_nx * ny * nz

    comm.Barrier()
    return running_sum_sq, local_rows


def dedalus_vds_source_mappings(input_path):
    """Return direct source-file mappings for a virtual Dedalus velocity task."""
    with h5py.File(input_path, "r") as hf:
        u_task = hf["tasks"]["u"]
        if not u_task.is_virtual:
            return None

        base_dir = os.path.dirname(os.path.abspath(input_path))
        mappings = []
        for source in u_task.virtual_sources():
            if source.dset_name != "tasks/u":
                continue
            if not source.src_space.is_regular_hyperslab() or not source.vspace.is_regular_hyperslab():
                raise ValueError(
                    "Dedalus VDS source mapping is not a regular hyperslab; "
                    "cannot safely import it directly."
                )
            src_start, src_span = regular_hyperslab_geometry(source.src_space)
            vds_start, vds_span = regular_hyperslab_geometry(source.vspace)
            mappings.append(
                {
                    "file_name": os.path.join(base_dir, source.file_name),
                    "dset_name": source.dset_name,
                    "src_start": src_start,
                    "src_span": src_span,
                    "vds_start": vds_start,
                    "vds_span": vds_span,
                }
            )

    return mappings


def bcast_dedalus_vds_source_mappings(input_path):
    """Broadcast VDS mapping discovery, raising discovery errors on every rank."""
    payload = None
    if rank == 0:
        try:
            payload = (dedalus_vds_source_mappings(input_path), None)
        except Exception as exc:
            payload = (None, repr(exc))
    mappings, error = comm.bcast(payload, root=0)
    if error is not None:
        raise ValueError(f"Failed to inspect Dedalus VDS source mappings: {error}")
    return mappings


def stream_dedalus_vds_sources_to_structured_h5(
    output_path,
    snapshot_info,
    mappings,
    x_block_size,
):
    """Import a Dedalus virtual dataset by reading the source piece files directly."""
    global_shape = snapshot_info["shape"]
    selected = snapshot_info["selected_index"]
    running_sum_sq = 0.0
    local_rows = 0

    if rank == 0 and os.path.exists(output_path):
        os.remove(output_path)
    comm.Barrier()

    with h5py.File(output_path, "w", driver="mpio", comm=comm) as out_hf:
        fields = out_hf.create_group("fields")
        output_dsets = (
            fields.create_dataset("vx", shape=global_shape, dtype="float64"),
            fields.create_dataset("vy", shape=global_shape, dtype="float64"),
            fields.create_dataset("vz", shape=global_shape, dtype="float64"),
        )

        for mapping_index in range(rank, len(mappings), size):
            mapping = mappings[mapping_index]
            src_start = mapping["src_start"]
            src_span = mapping["src_span"]
            vds_start = mapping["vds_start"]
            vds_span = mapping["vds_span"]

            source_write = src_start[0] + selected - vds_start[0]

            component_count = min(3, vds_span[1])
            if component_count < 3:
                raise ValueError(
                    f"Expected a VDS mapping with all 3 velocity components, got {component_count}."
                )

            x0, y0, z0 = vds_start[2], vds_start[3], vds_start[4]
            nx, ny, nz = vds_span[2], vds_span[3], vds_span[4]
            sx0, sy0, sz0 = src_start[2], src_start[3], src_start[4]

            with h5py.File(mapping["file_name"], "r") as source_hf:
                source_dset = source_hf[mapping["dset_name"]]
                if source_write < 0 or source_write >= source_dset.shape[0]:
                    continue
                for block_offset in range(0, nx, x_block_size):
                    block_nx = min(x_block_size, nx - block_offset)
                    buffer = np.empty((1, 1, block_nx, ny, nz), dtype=np.float64)
                    output_slab = np.s_[x0 + block_offset:x0 + block_offset + block_nx, y0:y0 + ny, z0:z0 + nz]
                    for component, output_dset in enumerate(output_dsets):
                        source_component = src_start[1] + component - vds_start[1]
                        source_dset.read_direct(
                            buffer,
                            source_sel=np.s_[
                                source_write:source_write + 1,
                                source_component:source_component + 1,
                                sx0 + block_offset:sx0 + block_offset + block_nx,
                                sy0:sy0 + ny,
                                sz0:sz0 + nz,
                            ],
                        )
                        block_values = buffer[0, 0]
                        output_dset[output_slab] = block_values
                        running_sum_sq += np.sum(block_values**2, dtype=np.float64)

                    local_rows += block_nx * ny * nz

    comm.Barrier()
    return running_sum_sq, local_rows


def import_dedalus_snapshot_to_structured_h5(input_path, output_path=None, write_index=-1, x_block_size=None):
    """Import one Dedalus velocity snapshot into the structured FFT-ready HDF5 schema."""
    snapshot_info = comm.bcast(
        dedalus_snapshot_info(input_path, write_index=write_index) if rank == 0 else None,
        root=0,
    )
    if output_path is None:
        output_path = dedalus_snapshot_output_path(input_path, snapshot_info)
    output_path = comm.bcast(output_path if rank == 0 else None, root=0)

    x_unique = snapshot_info["x"]
    y_unique = snapshot_info["y"]
    z_unique = snapshot_info["z"]
    nx = snapshot_info["shape"][0]
    x_ranges = split_axis(nx, size)
    x_start, x_stop = x_ranges[rank]
    x_block_size = resolve_dedalus_import_x_block_size(x_block_size)
    vds_mappings = bcast_dedalus_vds_source_mappings(input_path)

    if rank == 0:
        print(f"\nProcessing: {input_path}")
        print("  Input is Dedalus field-output HDF5.")
        print(
            f"  Selected write {snapshot_info['selected_index'] + 1}/{snapshot_info['num_writes']} "
            f"(write_number={snapshot_info['write_number']}, time={snapshot_info['sim_time']:.16e})"
        )
        if snapshot_info["cycle"] is not None:
            print(f"  Cycle: {snapshot_info['cycle']}")
        if h5py.get_config().mpi:
            if vds_mappings:
                print(
                    "  Detected virtual Dedalus velocity task; "
                    f"reading {len(vds_mappings)} source mappings directly."
                )
                print(f"  Source-file streaming x-block size: {x_block_size}")
            else:
                print(
                    "  Streaming Dedalus velocity task into structured HDF5 "
                    f"with x-block size {x_block_size}..."
                )
        else:
            print("  Reading Dedalus velocity task with independent serial HDF5 reads on each rank...")

    if h5py.get_config().mpi:
        if rank == 0:
            print("  Writing structured HDF5 in parallel as blocks are read...")
        if vds_mappings:
            running_sum_sq, local_rows = stream_dedalus_vds_sources_to_structured_h5(
                output_path,
                snapshot_info,
                vds_mappings,
                x_block_size,
            )
            import_mode = "direct_vds_sources"
        else:
            running_sum_sq, local_rows = stream_dedalus_snapshot_to_structured_h5(
                input_path,
                output_path,
                snapshot_info,
                x_start,
                x_stop,
                x_block_size,
            )
            import_mode = "streamed_read_direct"
        if rank == 0:
            write_structured_h5_metadata(
                output_path,
                x_unique,
                y_unique,
                z_unique,
                dedalus_header_lines(input_path, snapshot_info),
                step_number=(
                    str(snapshot_info["cycle"])
                    if snapshot_info["cycle"] is not None
                    else str(snapshot_info["write_number"])
                ),
                time_value=float(snapshot_info["sim_time"]),
                periodic_duplicate_last=False,
                extra_attrs={
                    "source_format": "dedalus_field_output_v1",
                    "source_file": os.path.abspath(input_path),
                    "source_write_index": int(snapshot_info["selected_index"]),
                    "source_write_number": int(snapshot_info["write_number"]),
                    "source_cycle": int(snapshot_info["cycle"]) if snapshot_info["cycle"] is not None else -1,
                    "source_handler_name": snapshot_info["handler_name"],
                    "source_set_number": int(snapshot_info["set_number"]),
                    "dedalus_import_x_block_size": int(x_block_size),
                    "dedalus_import_mode": import_mode,
                    "dedalus_vds_source_mappings": int(len(vds_mappings or [])),
                },
            )
        comm.Barrier()
    else:
        with open_h5_for_independent_read(input_path) as hf:
            u_task = hf["tasks"]["u"]
            local_vx = np.asarray(
                u_task[snapshot_info["selected_index"], 0, x_start:x_stop, :, :],
                dtype=np.float64,
            )
            local_vy = np.asarray(
                u_task[snapshot_info["selected_index"], 1, x_start:x_stop, :, :],
                dtype=np.float64,
            )
            local_vz = np.asarray(
                u_task[snapshot_info["selected_index"], 2, x_start:x_stop, :, :],
                dtype=np.float64,
            )

        running_sum_sq = np.sum(local_vx**2 + local_vy**2 + local_vz**2, dtype=np.float64)
        local_rows = int(local_vx.size)

        if rank == 0:
            print("  h5py MPI support is unavailable; falling back to serial HDF5 assembly on rank 0...")

        write_structured_h5(
            output_path,
            x_unique,
            y_unique,
            z_unique,
            (x_start, x_stop),
            local_vx,
            local_vy,
            local_vz,
            dedalus_header_lines(input_path, snapshot_info),
            step_number=(
                str(snapshot_info["cycle"])
                if snapshot_info["cycle"] is not None
                else str(snapshot_info["write_number"])
            ),
            time_value=float(snapshot_info["sim_time"]),
            periodic_duplicate_last=False,
            extra_attrs={
                "source_format": "dedalus_field_output_v1",
                "source_file": os.path.abspath(input_path),
                "source_write_index": int(snapshot_info["selected_index"]),
                "source_write_number": int(snapshot_info["write_number"]),
                "source_cycle": int(snapshot_info["cycle"]) if snapshot_info["cycle"] is not None else -1,
                "source_handler_name": snapshot_info["handler_name"],
                "source_set_number": int(snapshot_info["set_number"]),
                "dedalus_import_mode": "whole_rank_slab",
            },
        )
        del local_vx, local_vy, local_vz

    global_points = int(np.prod(snapshot_info["shape"]))
    global_sum_sq = comm.reduce(running_sum_sq, op=MPI.SUM, root=0)
    total_rows = comm.reduce(local_rows, op=MPI.SUM, root=0)

    output_tke = None
    if rank == 0:
        if total_rows != global_points:
            raise RuntimeError(
                "Dedalus snapshot import wrote an unexpected number of rows: "
                f"expected {global_points}, wrote {total_rows}."
            )
        output_tke = 0.5 * (global_sum_sq / global_points)
        print(f"  Import complete. Total grid points: {global_points}")
    output_tke = comm.bcast(output_tke, root=0)

    verified_tke, verified_rows = calculate_file_tke_parallel(output_path)
    if verified_tke is None:
        raise RuntimeError(f"Verification failed for imported Dedalus snapshot {output_path}")

    success = False
    if rank == 0:
        print(f"  Original TKE:       {output_tke:.16f}")
        print(f"  Reconstructed TKE:  {verified_tke:.16f}")
        print(f"  Original Rows:      {global_points}")
        print(f"  Reconstructed Rows: {verified_rows}")

        tke_match = np.isclose(output_tke, verified_tke, atol=1e-12)
        row_match = global_points == verified_rows
        if tke_match and row_match:
            print("  SUCCESS: TKE and Row Counts match.")
            print_storage_stats(input_path, output_path)
            success = True
        else:
            raise RuntimeError(
                "Dedalus snapshot import integrity check failed: "
                f"TKE match={tke_match}, row match={row_match}."
            )

    success = comm.bcast(success, root=0)
    if not success:
        raise RuntimeError(f"Failed to import Dedalus snapshot from {input_path}")
    return output_path


def import_all_dedalus_snapshots_to_structured_h5(input_path, x_block_size=None):
    """Import all writes from a multi-write Dedalus field-output file.

    Returns a list of structured HDF5 paths, one per write.
    """
    num_writes = None
    if rank == 0:
        with h5py.File(input_path, "r") as hf:
            num_writes = int(hf["tasks"]["u"].shape[0])
    num_writes = comm.bcast(num_writes, root=0)

    if rank == 0 and num_writes > 1:
        print(f"\n  Dedalus file contains {num_writes} writes; importing all snapshots...")

    output_paths = []
    for write_idx in range(num_writes):
        output_path = import_dedalus_snapshot_to_structured_h5(
            input_path,
            write_index=write_idx,
            x_block_size=x_block_size,
        )
        output_paths.append(output_path)

    return output_paths


def write_scalar_field_to_h5(
    h5_path,
    field_name,
    display_name,
    local_x_bounds,
    local_scalar,
    source_path,
):
    """Append one scalar field dataset to an existing structured HDF5 file."""
    x_unique, y_unique, z_unique = read_structured_grid(h5_path)
    global_shape = (len(x_unique), len(y_unique), len(z_unique))
    x_start, x_stop = local_x_bounds

    if rank == 0:
        with h5py.File(h5_path, "a") as hf:
            fields = hf["fields"]
            if field_name in fields:
                del fields[field_name]
    comm.Barrier()

    if h5py.get_config().mpi:
        with h5py.File(h5_path, "r+", driver="mpio", comm=comm) as hf:
            dataset = hf["fields"].create_dataset(field_name, shape=global_shape, dtype="float64")
            if x_stop > x_start:
                dataset[slice(x_start, x_stop), :, :] = local_scalar
    else:
        gathered = comm.gather((x_start, x_stop, local_scalar), root=0)
        if rank == 0:
            with h5py.File(h5_path, "a") as hf:
                dataset = hf["fields"].create_dataset(field_name, shape=global_shape, dtype="float64")
                for slab_start, slab_stop, slab_values in gathered:
                    if slab_stop > slab_start:
                        dataset[slice(slab_start, slab_stop), :, :] = slab_values

    comm.Barrier()

    if rank == 0:
        with h5py.File(h5_path, "a") as hf:
            dataset = hf["fields"][field_name]
            dataset.attrs["field_kind"] = "scalar"
            dataset.attrs["display_name"] = display_name
            dataset.attrs["plot_label"] = display_name
            dataset.attrs["source_path"] = os.path.abspath(source_path)
            if str(source_path).lower().endswith(".txt"):
                dataset.attrs["source_txt"] = os.path.abspath(source_path)
            elif str(source_path).lower().endswith(".h5"):
                dataset.attrs["source_h5"] = os.path.abspath(source_path)


def write_scalar_structured_h5(
    h5_path,
    x_unique,
    y_unique,
    z_unique,
    field_name,
    display_name,
    local_x_bounds,
    local_scalar,
    header_lines,
    source_txt_path,
):
    """Write one scalar sampled-data TXT file to its own structured HDF5 file."""
    global_shape = (len(x_unique), len(y_unique), len(z_unique))
    x_start, x_stop = local_x_bounds

    if rank == 0 and os.path.exists(h5_path):
        os.remove(h5_path)
    comm.Barrier()

    if h5py.get_config().mpi:
        with h5py.File(h5_path, "w", driver="mpio", comm=comm) as hf:
            fields = hf.create_group("fields")
            scalar_dset = fields.create_dataset(field_name, shape=global_shape, dtype="float64")
            if x_stop > x_start:
                scalar_dset[slice(x_start, x_stop), :, :] = local_scalar
    else:
        gathered = comm.gather((x_start, x_stop, local_scalar), root=0)
        if rank == 0:
            with h5py.File(h5_path, "w") as hf:
                fields = hf.create_group("fields")
                scalar_dset = fields.create_dataset(field_name, shape=global_shape, dtype="float64")
                for slab_start, slab_stop, slab_values in gathered:
                    if slab_stop > slab_start:
                        scalar_dset[slice(slab_start, slab_stop), :, :] = slab_values

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

            dataset = hf["fields"][field_name]
            dataset.attrs["field_kind"] = "scalar"
            dataset.attrs["display_name"] = display_name
            dataset.attrs["plot_label"] = display_name
            dataset.attrs["source_txt"] = os.path.abspath(source_txt_path)

            hf.attrs["schema"] = "structured_scalar_v1"
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
            hf.attrs["source_txt"] = os.path.abspath(source_txt_path)
            hf.attrs["field_name"] = field_name
            hf.attrs["display_name"] = display_name


def scalar_h5_output_path(txt_path):
    """Return the standalone HDF5 output path for one scalar TXT file."""
    base, _ = os.path.splitext(txt_path)
    return base + ".h5"


def read_scalar_field_xslab_from_h5(h5_path, field_name, grid_shape):
    """Read one rank-local x-slab from a standalone structured scalar HDF5 file."""
    nx, ny, nz = grid_shape
    x_start, x_stop = split_axis(nx, size)[rank]
    local_nx = x_stop - x_start
    local_scalar = np.zeros((local_nx, ny, nz), dtype=np.float64)

    if local_nx > 0:
        with open_h5_for_independent_read(h5_path) as hf:
            local_scalar[...] = np.asarray(hf["fields"][field_name][slice(x_start, x_stop), :, :], dtype=np.float64)

    return (x_start, x_stop), local_scalar


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


def append_scalar_txt_to_h5_parallel(txt_path, h5_path, scalar_h5_path=None):
    """Append one scalar sampled-data TXT file into an existing structured HDF5 file."""
    exists = comm.bcast(os.path.exists(txt_path) if rank == 0 else False, root=0)
    if not exists:
        raise FileNotFoundError(txt_path)

    if rank == 0:
        print(f"\nAdding scalar field: {txt_path}")
        header_lines, skip_count = get_txt_header(txt_path)
        if header_lines is not None:
            print("  Building scalar chunk index (rank 0 scans file once)...")
            chunk_index = build_chunk_index(txt_path, skip_count)
            total_rows = sum(n for _, n in chunk_index)
            field_name, display_name = parse_sampled_data_field_name(
                header_lines,
                os.path.splitext(os.path.basename(txt_path))[0],
            )
            print(f"  Parsed scalar name: {display_name} -> dataset '{field_name}'")
            print(f"  {total_rows} rows | {len(chunk_index)} chunks | {size} MPI ranks")
        else:
            chunk_index = None
            total_rows = 0
            field_name = None
            display_name = None
    else:
        header_lines = None
        chunk_index = None
        total_rows = 0
        field_name = None
        display_name = None

    header_lines = comm.bcast(header_lines, root=0)
    chunk_index = comm.bcast(chunk_index, root=0)
    total_rows = comm.bcast(total_rows, root=0)
    field_name = comm.bcast(field_name, root=0)
    display_name = comm.bcast(display_name, root=0)

    if header_lines is None or not chunk_index:
        raise RuntimeError(f"Failed to read scalar field header from {txt_path}")

    if field_name in {"vx", "vy", "vz"}:
        raise ValueError(f"Scalar dataset name '{field_name}' conflicts with required velocity fields.")

    if rank == 0:
        print("  Discovering scalar field grid in parallel...")
    txt_x, txt_y, txt_z, dx, dy, dz = discover_grid_from_chunks(txt_path, chunk_index)
    h5_x, h5_y, h5_z = read_structured_grid(h5_path)
    validate_matching_structured_grid(txt_path, h5_path, (txt_x, txt_y, txt_z), (h5_x, h5_y, h5_z))

    expected_rows = len(txt_x) * len(txt_y) * len(txt_z)
    if expected_rows != total_rows:
        raise ValueError(
            f"Scalar field grid size ({expected_rows}) does not match text row count ({total_rows}) for {txt_path}."
        )

    if rank == 0:
        print("  Redistributing scalar rows into FFT-ready x-slabs...")
    local_x_bounds, local_scalar, local_rows = redistribute_scalar_to_xslabs(
        txt_path,
        chunk_index,
        txt_x,
        txt_y,
        txt_z,
        dx,
        dy,
        dz,
    )

    if rank == 0:
        if h5py.get_config().mpi:
            print("  Writing scalar field into structured HDF5 in parallel...")
        else:
            print("  h5py MPI support is unavailable; falling back to serial scalar HDF5 assembly on rank 0...")

    write_scalar_field_to_h5(
        h5_path,
        field_name,
        display_name,
        local_x_bounds,
        local_scalar,
        txt_path,
    )

    if scalar_h5_path is not None:
        if rank == 0:
            if h5py.get_config().mpi:
                print(f"  Writing standalone scalar HDF5: {scalar_h5_path}")
            else:
                print(f"  Writing standalone scalar HDF5 on rank 0: {scalar_h5_path}")
        write_scalar_structured_h5(
            scalar_h5_path,
            txt_x,
            txt_y,
            txt_z,
            field_name,
            display_name,
            local_x_bounds,
            local_scalar,
            header_lines,
            txt_path,
        )

    total_written_rows = comm.reduce(local_rows, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"  Scalar append complete. Total rows: {total_written_rows}")
    total_written_rows = comm.bcast(total_written_rows, root=0)
    if total_written_rows != total_rows:
        raise ValueError(
            f"Scalar field row count mismatch for {txt_path}: expected {total_rows}, wrote {total_written_rows}."
        )

    return field_name


def append_scalar_h5_to_h5_parallel(scalar_h5_path, h5_path):
    """Append one standalone structured scalar HDF5 file into an existing structured HDF5 file."""
    exists = comm.bcast(os.path.exists(scalar_h5_path) if rank == 0 else False, root=0)
    if not exists:
        raise FileNotFoundError(scalar_h5_path)

    if rank == 0:
        print(f"\nAdding scalar field: {scalar_h5_path}")
        print("  Reading standalone structured scalar HDF5 metadata...")

    field_name, display_name, scalar_x, scalar_y, scalar_z, total_rows = read_scalar_structured_h5_metadata(
        scalar_h5_path
    )
    if field_name in {"vx", "vy", "vz"}:
        raise ValueError(f"Scalar dataset name '{field_name}' conflicts with required velocity fields.")

    h5_x, h5_y, h5_z = read_structured_grid(h5_path)
    validate_matching_structured_grid(scalar_h5_path, h5_path, (scalar_x, scalar_y, scalar_z), (h5_x, h5_y, h5_z))

    if rank == 0:
        print(f"  Parsed scalar name: {display_name} -> dataset '{field_name}'")
        print(f"  {total_rows} rows | {size} MPI ranks")
        print("  Reading scalar HDF5 x-slabs directly...")
    local_x_bounds, local_scalar = read_scalar_field_xslab_from_h5(
        scalar_h5_path,
        field_name,
        (len(scalar_x), len(scalar_y), len(scalar_z)),
    )
    local_rows = int(local_scalar.size)

    if rank == 0:
        if h5py.get_config().mpi:
            print("  Writing scalar field into structured HDF5 in parallel...")
        else:
            print("  h5py MPI support is unavailable; falling back to serial scalar HDF5 assembly on rank 0...")

    write_scalar_field_to_h5(
        h5_path,
        field_name,
        display_name,
        local_x_bounds,
        local_scalar,
        scalar_h5_path,
    )

    total_written_rows = comm.reduce(local_rows, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"  Scalar append complete. Total rows: {total_written_rows}")
    total_written_rows = comm.bcast(total_written_rows, root=0)
    if total_written_rows != total_rows:
        raise ValueError(
            f"Scalar field row count mismatch for {scalar_h5_path}: expected {total_rows}, wrote {total_written_rows}."
        )

    return field_name


def append_scalar_fields_to_h5(paths, h5_path, create_scalar_h5=False):
    """Append one or more scalar TXT/HDF5 inputs into an existing structured HDF5 file."""
    added_fields = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            scalar_h5_path = scalar_h5_output_path(path) if create_scalar_h5 else None
            added_fields.append(append_scalar_txt_to_h5_parallel(path, h5_path, scalar_h5_path=scalar_h5_path))
        elif ext == ".h5":
            added_fields.append(append_scalar_h5_to_h5_parallel(path, h5_path))
        else:
            raise ValueError(f"Unsupported scalar input extension '{ext}'. Expected .txt or .h5.")
    return added_fields


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
                    ny, nz = vx_dset.shape[1], vx_dset.shape[2]
                    x_block = max(1, CHUNK_SIZE // max(ny * nz, 1))
                    for block_start in range(x_start, x_stop, x_block):
                        block_stop = min(block_start + x_block, x_stop)
                        block_shape = (block_stop - block_start, ny, nz)
                        buffer = np.empty(block_shape, dtype=np.float64)
                        block_sum_sq = 0.0
                        source_sel = np.s_[block_start:block_stop, :, :]
                        for dset in (vx_dset, vy_dset, vz_dset):
                            dset.read_direct(buffer, source_sel=source_sel)
                            block_sum_sq += np.sum(buffer**2, dtype=np.float64)
                        running_sum_sq += block_sum_sq
                        local_rows += buffer.size
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
