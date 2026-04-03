#!/usr/bin/env python3
"""
Visualize a center slice of the velocity field using MPI-assisted I/O plus yt.

Supported inputs:
  - SampledData text files
  - Structured HDF5 files written by convert_txt_to_hdf5.py

Examples:
  mpirun -n 4 python visualize_velocity_yt.py data/SampledData0.txt --axis z --field vx
  mpirun -n 4 python visualize_velocity_yt.py data/SampledData0.h5 --axis y --field vz
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yt
from mpi4py import rc

rc.initialize = False
rc.finalize = False

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

import convert_txt_to_hdf5 as parallel_io


FIELD_MAP = {
    "vx": ("gas", "velocity_x"),
    "vy": ("gas", "velocity_y"),
    "vz": ("gas", "velocity_z"),
    "velocity_x": ("gas", "velocity_x"),
    "velocity_y": ("gas", "velocity_y"),
    "velocity_z": ("gas", "velocity_z"),
}


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_header_metadata_from_lines(lines):
    """Extract step and time from a text header."""
    step_number = "unknown"
    time_value = 0.0
    for line in lines:
        step_match = re.search(r"(?:Step|Cycle)\s*[:=]?\s*(\d+)", line)
        if step_match:
            step_number = step_match.group(1)
        time_match = re.search(r"Time\s*[:=]?\s*([0-9.eE+-]+)", line)
        if time_match:
            time_value = float(time_match.group(1))
    return step_number, time_value


def canonical_field_name(field_name):
    """Return the yt field tuple and the source velocity component name."""
    if field_name not in FIELD_MAP:
        raise ValueError(f"Unsupported field '{field_name}'.")
    return FIELD_MAP[field_name]


def field_component_key(field_name):
    """Map a CLI field name to the raw component key stored in arrays."""
    yt_field = canonical_field_name(field_name)[1]
    return {
        "velocity_x": "vx",
        "velocity_y": "vy",
        "velocity_z": "vz",
    }[yt_field]


def gather_global_slabs(local_bounds, local_vx, local_vy, local_vz, shape):
    """Gather rank-local x-slabs to rank 0 and assemble global arrays."""
    gathered = comm.gather((local_bounds, local_vx, local_vy, local_vz), root=0)
    if rank != 0:
        return None, None, None

    global_vx = np.zeros(shape, dtype=np.float64)
    global_vy = np.zeros(shape, dtype=np.float64)
    global_vz = np.zeros(shape, dtype=np.float64)
    for (x_start, x_stop), slab_vx, slab_vy, slab_vz in gathered:
        if x_stop > x_start:
            slab = slice(x_start, x_stop)
            global_vx[slab, :, :] = slab_vx
            global_vy[slab, :, :] = slab_vy
            global_vz[slab, :, :] = slab_vz
    return global_vx, global_vy, global_vz


def load_structured_h5_parallel(filepath):
    """Read one structured HDF5 file in parallel and assemble on rank 0."""
    with h5py.File(filepath, "r") as hf:
        x_full = np.asarray(hf["grid"]["x"][:], dtype=np.float64)
        y_full = np.asarray(hf["grid"]["y"][:], dtype=np.float64)
        z_full = np.asarray(hf["grid"]["z"][:], dtype=np.float64)
        periodic_duplicate_last = bool(hf.attrs.get("periodic_duplicate_last", True))
        step = str(hf.attrs.get("step", "unknown"))
        time_val = float(hf.attrs.get("time", 0.0))

    if periodic_duplicate_last and len(x_full) > 1 and len(y_full) > 1 and len(z_full) > 1:
        x_coords = x_full[:-1]
        y_coords = y_full[:-1]
        z_coords = z_full[:-1]
    else:
        x_coords = x_full
        y_coords = y_full
        z_coords = z_full

    shape = (len(x_coords), len(y_coords), len(z_coords))
    x_ranges = parallel_io.split_axis(shape[0], size)
    x_start, x_stop = x_ranges[rank]

    if h5py.get_config().mpi:
        with h5py.File(filepath, "r", driver="mpio", comm=comm) as hf:
            local_vx = np.asarray(hf["fields"]["vx"][x_start:x_stop, : shape[1], : shape[2]], dtype=np.float64)
            local_vy = np.asarray(hf["fields"]["vy"][x_start:x_stop, : shape[1], : shape[2]], dtype=np.float64)
            local_vz = np.asarray(hf["fields"]["vz"][x_start:x_stop, : shape[1], : shape[2]], dtype=np.float64)
    else:
        with h5py.File(filepath, "r") as hf:
            local_vx = np.asarray(hf["fields"]["vx"][x_start:x_stop, : shape[1], : shape[2]], dtype=np.float64)
            local_vy = np.asarray(hf["fields"]["vy"][x_start:x_stop, : shape[1], : shape[2]], dtype=np.float64)
            local_vz = np.asarray(hf["fields"]["vz"][x_start:x_stop, : shape[1], : shape[2]], dtype=np.float64)

    global_vx, global_vy, global_vz = gather_global_slabs((x_start, x_stop), local_vx, local_vy, local_vz, shape)
    return {
        "x": x_coords,
        "y": y_coords,
        "z": z_coords,
        "vx": global_vx,
        "vy": global_vy,
        "vz": global_vz,
        "step": step,
        "time": time_val,
    }


def load_txt_parallel(filepath):
    """Read one text file in parallel using the converter's distributed loading pattern."""
    header_lines = None
    chunk_index = None
    total_rows = 0

    if rank == 0:
        header_lines, skip_count = parallel_io.get_txt_header(filepath)
        chunk_index = parallel_io.build_chunk_index(filepath, skip_count)
        total_rows = sum(nrows for _, nrows in chunk_index)

    header_lines = comm.bcast(header_lines, root=0)
    chunk_index = comm.bcast(chunk_index, root=0)
    total_rows = comm.bcast(total_rows, root=0)

    if header_lines is None or not chunk_index:
        raise ValueError(f"Unable to load text file: {filepath}")

    x_unique, y_unique, z_unique, dx, dy, dz = parallel_io.discover_grid_from_chunks(filepath, chunk_index)
    expected_rows = len(x_unique) * len(y_unique) * len(z_unique)
    if rank == 0 and expected_rows != total_rows:
        raise ValueError(
            f"Structured grid size ({expected_rows}) does not match text row count ({total_rows})."
        )

    local_x_bounds, local_vx, local_vy, local_vz, _ = parallel_io.redistribute_to_xslabs(
        filepath, chunk_index, x_unique, y_unique, z_unique, dx, dy, dz
    )
    global_vx, global_vy, global_vz = gather_global_slabs(
        local_x_bounds,
        local_vx,
        local_vy,
        local_vz,
        (len(x_unique), len(y_unique), len(z_unique)),
    )

    step, time_val = parse_header_metadata_from_lines(header_lines)
    return {
        "x": x_unique,
        "y": y_unique,
        "z": z_unique,
        "vx": global_vx,
        "vy": global_vy,
        "vz": global_vz,
        "step": step,
        "time": time_val,
    }


def load_data_parallel(filepath):
    """Load a TXT or structured HDF5 file with MPI-assisted reads."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        return load_txt_parallel(filepath)
    if ext == ".h5":
        return load_structured_h5_parallel(filepath)
    raise ValueError(f"Unsupported file extension: {ext}")


def build_yt_dataset(global_data):
    """Create a yt uniform-grid dataset on rank 0."""
    x = global_data["x"]
    y = global_data["y"]
    z = global_data["z"]

    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    dz = float(z[1] - z[0]) if len(z) > 1 else 1.0

    bbox = np.array(
        [
            [x[0], x[-1] + dx],
            [y[0], y[-1] + dy],
            [z[0], z[-1] + dz],
        ],
        dtype=np.float64,
    )

    data = {
        ("gas", "velocity_x"): (global_data["vx"], "dimensionless"),
        ("gas", "velocity_y"): (global_data["vy"], "dimensionless"),
        ("gas", "velocity_z"): (global_data["vz"], "dimensionless"),
    }
    return yt.load_uniform_grid(data, global_data["vx"].shape, bbox=bbox, length_unit="m", nprocs=1)


def center_for_axis(ds, axis):
    """Return the 3D center coordinate for an axis-aligned slice."""
    center = np.array(ds.domain_center, dtype=np.float64)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    center[axis_index] = 0.5 * float(ds.domain_left_edge[axis_index] + ds.domain_right_edge[axis_index])
    return center


def output_name(data_file, field_name, axis):
    base, _ = os.path.splitext(data_file)
    return f"{base}_yt_{field_name}_center_slice_{axis}.png"


def main():
    parser = argparse.ArgumentParser(description="Visualize a center slice of the velocity field with yt")
    parser.add_argument("data_file", help="Path to a SampledData .txt or structured .h5 file")
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"], help="Slice normal axis")
    parser.add_argument(
        "--field",
        default="vx",
        choices=sorted(FIELD_MAP),
        help="Velocity component to plot",
    )
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib/yt colormap")
    parser.add_argument("--width", type=float, default=None, help="Optional plot width in domain units")
    parser.add_argument("--output", default=None, help="Optional output PNG path")
    parser.add_argument("--plot", action="store_true", help="Also display the plot on rank 0 after saving")
    args = parser.parse_args()

    if rank == 0:
        print(f"Loading {args.data_file} with {size} MPI ranks...")
    global_data = load_data_parallel(args.data_file)

    if rank == 0:
        ds = build_yt_dataset(global_data)
        field = canonical_field_name(args.field)
        center = center_for_axis(ds, args.axis)

        print(f"Step: {global_data['step']}, Time: {global_data['time']}")
        print(f"Domain dimensions: {tuple(int(v) for v in ds.domain_dimensions)}")
        print(f"Center slice axis: {args.axis}, center: {tuple(float(v) for v in center)}")

        slc = yt.SlicePlot(ds, args.axis, field, center=center)
        slc.set_cmap(field, args.cmap)
        slc.annotate_title(
            f"{field[1]} center slice, axis={args.axis}, step={global_data['step']}, time={global_data['time']:.6g}"
        )
        if args.width is not None:
            slc.set_width(args.width)

        out = args.output or output_name(args.data_file, field[1], args.axis)
        slc.save(out)
        print(f"Saved: {out}")
        if args.plot:
            slc._setup_plots()
            plt.show()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 0
    finally:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()

    sys.exit(exit_code)
