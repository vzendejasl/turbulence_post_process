"""Helpers for persisting and reloading saved 2D slice data."""

from __future__ import annotations

import os

import h5py
import numpy as np


SLICE_SCHEMA_VERSION = 1

PLANE_AXES = {
    "x": ("z", "y"),
    "y": ("z", "x"),
    "z": ("y", "x"),
}

PLANE_NAMES = {
    "x": "yz",
    "y": "zx",
    "z": "xy",
}


def default_slice_data_output(data_file):
    """Return the default combined slice-data HDF5 path for one input file."""
    directory = os.path.dirname(os.path.abspath(data_file))
    output_dir = os.path.join(directory, "slice_data")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(data_file))[0]
    return os.path.join(output_dir, f"{base}_slices.h5")


def storage_slice_tag(axis, slice_tag):
    """Return the stored HDF5 slice tag, namespaced by plane orientation when needed."""
    if slice_tag in {"xy_center", "xy_face", "yz_face", "zx_face"}:
        return slice_tag
    if slice_tag.startswith(f"{PLANE_NAMES[axis]}_"):
        return slice_tag
    return f"{PLANE_NAMES[axis]}_{slice_tag}"


def plane_shape(shape, axis):
    """Return the global 2D plane shape for one slice orientation."""
    nx, ny, nz = shape
    if axis == "x":
        return ny, nz
    if axis == "y":
        return nx, nz
    return nx, ny


def plane_coordinate_arrays(meta, axis):
    """Return the stored horizontal/vertical coordinate arrays for one plane."""
    if axis == "x":
        return meta["z"], meta["y"]
    if axis == "y":
        return meta["z"], meta["x"]
    return meta["y"], meta["x"]


def plane_extent_from_arrays(horizontal_coords, vertical_coords):
    """Return imshow extent bounds from 1D horizontal/vertical coordinates."""
    horizontal_coords = np.asarray(horizontal_coords, dtype=np.float64)
    vertical_coords = np.asarray(vertical_coords, dtype=np.float64)

    if len(horizontal_coords) == 1:
        dx = 1.0
    else:
        dx = float(horizontal_coords[1] - horizontal_coords[0])
    if len(vertical_coords) == 1:
        dy = 1.0
    else:
        dy = float(vertical_coords[1] - vertical_coords[0])

    return [
        float(horizontal_coords[0]),
        float(horizontal_coords[-1] + dx),
        float(vertical_coords[0]),
        float(vertical_coords[-1] + dy),
    ]


def open_h5_for_parallel_write(filepath, comm):
    """Open an HDF5 file for parallel writes when MPI-enabled h5py is available."""
    if h5py.get_config().mpi:
        return h5py.File(filepath, "r+", driver="mpio", comm=comm)
    return h5py.File(filepath, "r+")


def initialize_slice_data_file(
    filepath,
    meta,
    field_specs,
    requests,
    source_file,
    source_h5,
    backend_name,
):
    """Create a fresh combined slice-data HDF5 file for one processed input."""
    output_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(filepath, "w") as hf:
        hf.attrs["schema"] = "turbulence_post_process.slice_data"
        hf.attrs["schema_version"] = SLICE_SCHEMA_VERSION
        hf.attrs["source_file"] = os.path.abspath(source_file)
        hf.attrs["source_h5"] = os.path.abspath(source_h5)
        hf.attrs["step"] = str(meta["step"])
        hf.attrs["time"] = float(meta["time"])
        hf.attrs["backend"] = backend_name
        hf.attrs["grid_shape"] = np.asarray(meta["shape"], dtype=np.int64)

        grid_group = hf.create_group("grid")
        grid_group.create_dataset("x", data=np.asarray(meta["x"], dtype=np.float64))
        grid_group.create_dataset("y", data=np.asarray(meta["y"], dtype=np.float64))
        grid_group.create_dataset("z", data=np.asarray(meta["z"], dtype=np.float64))

        slices_group = hf.create_group("slices")
        for dataset_name, field_label, latex_label, field_family in field_specs:
            field_group = slices_group.create_group(field_label)
            field_group.attrs["field_name"] = field_label
            field_group.attrs["source_dataset"] = dataset_name
            field_group.attrs["plot_label"] = latex_label
            field_group.attrs["field_family"] = field_family

            for axis_name, plane_index, slice_tag in requests:
                stored_tag = storage_slice_tag(axis_name, slice_tag)
                slice_group = field_group.require_group(stored_tag)
                horizontal_name, vertical_name = PLANE_AXES[axis_name]
                horizontal_coords, vertical_coords = plane_coordinate_arrays(meta, axis_name)
                slice_group.attrs["axis"] = axis_name
                slice_group.attrs["slice_tag"] = stored_tag
                slice_group.attrs["plane_index"] = int(plane_index)
                slice_group.attrs["plane_coord"] = float(meta[axis_name][plane_index])
                slice_group.attrs["horizontal_axis"] = horizontal_name
                slice_group.attrs["vertical_axis"] = vertical_name
                slice_group.attrs["field_name"] = field_label
                slice_group.attrs["source_dataset"] = dataset_name
                slice_group.attrs["plot_label"] = latex_label
                slice_group.attrs["field_family"] = field_family
                slice_group.attrs["step"] = str(meta["step"])
                slice_group.attrs["time"] = float(meta["time"])
                slice_group.attrs["source_file"] = os.path.abspath(source_file)
                slice_group.attrs["source_h5"] = os.path.abspath(source_h5)
                slice_group.attrs["backend"] = backend_name
                slice_group.attrs["plane_shape"] = np.asarray(plane_shape(meta["shape"], axis_name), dtype=np.int64)
                if "values" not in slice_group:
                    slice_group.create_dataset("values", shape=plane_shape(meta["shape"], axis_name), dtype=np.float64)
                if "coord_horizontal" not in slice_group:
                    slice_group.create_dataset("coord_horizontal", data=np.asarray(horizontal_coords, dtype=np.float64))
                if "coord_vertical" not in slice_group:
                    slice_group.create_dataset("coord_vertical", data=np.asarray(vertical_coords, dtype=np.float64))


def write_slice_plane_parallel(filepath, field_label, slice_tag, axis, local_bounds, local_plane, comm):
    """Write one distributed plane into the combined HDF5 slice file."""
    with open_h5_for_parallel_write(filepath, comm) as hf:
        dataset = hf["slices"][field_label][slice_tag]["values"]
        if local_plane is None or local_bounds is None:
            return

        if axis == "x":
            y0, y1, z0, z1 = local_bounds
            dataset[y0:y1, z0:z1] = local_plane
            return
        if axis == "y":
            x0, x1, z0, z1 = local_bounds
            dataset[x0:x1, z0:z1] = local_plane
            return

        x0, x1, y0, y1 = local_bounds
        dataset[x0:x1, y0:y1] = local_plane


def write_slice_plane_serial(filepath, field_label, slice_tag, plane):
    """Write one full plane from rank 0 when MPI-enabled HDF5 is unavailable."""
    with h5py.File(filepath, "r+") as hf:
        hf["slices"][field_label][slice_tag]["values"][:] = np.asarray(plane, dtype=np.float64)


def list_available_slices(filepath):
    """Return summary metadata for a combined slice-data HDF5 file."""
    with h5py.File(filepath, "r") as hf:
        summary = {
            "source_file": str(hf.attrs.get("source_file", "")),
            "source_h5": str(hf.attrs.get("source_h5", "")),
            "step": str(hf.attrs.get("step", "unknown")),
            "time": float(hf.attrs.get("time", 0.0)),
            "grid_shape": tuple(int(value) for value in hf.attrs.get("grid_shape", ())),
            "fields": {},
        }

        slices_group = hf["slices"]
        for field_name in sorted(slices_group.keys()):
            field_group = slices_group[field_name]
            summary["fields"][field_name] = sorted(field_group.keys())

    return summary


def load_saved_slice(filepath, field_name, slice_tag):
    """Load one saved slice plane plus metadata from a combined slice-data file."""
    with h5py.File(filepath, "r") as hf:
        slice_group = hf["slices"][field_name][slice_tag]
        return {
            "values": np.asarray(slice_group["values"][:], dtype=np.float64),
            "coord_horizontal": np.asarray(slice_group["coord_horizontal"][:], dtype=np.float64),
            "coord_vertical": np.asarray(slice_group["coord_vertical"][:], dtype=np.float64),
            "attrs": {key: slice_group.attrs[key] for key in slice_group.attrs},
        }
