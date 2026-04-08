"""Scalable slice-plot workflow for structured velocity HDF5 files."""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
from mpi4py import MPI

from postprocess_fft.io import read_structured_local_fields
from postprocess_fft.layout import box_shape
from postprocess_fft.layout import box_slices
from postprocess_fft.layout import build_boxes
from postprocess_fft.layout import choose_proc_grid
from postprocess_fft.transform import backward_field
from postprocess_fft.transform import forward_field
from postprocess_fft.transform import get_backend
from postprocess_fft.transform import local_wavenumber_mesh
from postprocess_fft.common import heffte
from postprocess_lib.prepare import ensure_structured_h5
from postprocess_vis.slice_data import default_slice_data_output
from postprocess_vis.slice_data import initialize_slice_data_file
from postprocess_vis.slice_data import storage_slice_tag
from postprocess_vis.slice_data import write_slice_plane_parallel
from postprocess_vis.slice_data import write_slice_plane_serial


BUILTIN_FIELD_MAP = {
    "velocity_magnitude": ("velocity_magnitude", "velocity_magnitude", r"$|\mathbf{u}|$", "velocity"),
    "vx": ("vx", "vx", r"$u_1$", "velocity"),
    "vy": ("vy", "vy", r"$u_2$", "velocity"),
    "vz": ("vz", "vz", r"$u_3$", "velocity"),
    "vorticity_magnitude": ("vorticity_magnitude", "vorticity_magnitude", r"$|\boldsymbol{\omega}|$", "vorticity"),
    "wx": ("omega_x", "wx", r"$\omega_1$", "vorticity"),
    "wy": ("omega_y", "wy", r"$\omega_2$", "vorticity"),
    "wz": ("omega_z", "wz", r"$\omega_3$", "vorticity"),
}

PLANE_NAMES = {
    "x": "yz",
    "y": "zx",
    "z": "xy",
}

PLANE_AXES = {
    "x": ("z", "y"),
    "y": ("z", "x"),
    "z": ("y", "x"),
}


def log_rank0(rank, message):
    """Print one progress message from rank 0 and flush immediately."""
    if rank == 0:
        print(message, flush=True)


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


def canonical_field_name(field_name):
    """Return the HDF5 field name and human-readable label."""
    if field_name not in BUILTIN_FIELD_MAP:
        raise ValueError(f"Unsupported field '{field_name}'.")
    return BUILTIN_FIELD_MAP[field_name]


def available_field_specs(filepath):
    """Return the built-in and dataset-backed field specs available in one HDF5 file."""
    field_specs = dict(BUILTIN_FIELD_MAP)
    with h5py.File(filepath, "r") as hf:
        fields = hf["fields"]
        for dataset_name in sorted(fields.keys()):
            if dataset_name in {"vx", "vy", "vz"}:
                continue
            if dataset_name in field_specs:
                continue
            dataset = fields[dataset_name]
            display_name = str(dataset.attrs.get("display_name", dataset_name))
            plot_label = str(dataset.attrs.get("plot_label", display_name))
            field_specs[dataset_name] = (dataset_name, dataset_name, plot_label, "scalar")
    return field_specs


def _yt_font_style():
    return {
        "family": "serif",
        "size": 20,
    }


def read_grid_metadata(filepath):
    """Read grid coordinates and metadata from one structured HDF5 file."""
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

    return {
        "x": x_coords,
        "y": y_coords,
        "z": z_coords,
        "shape": (len(x_coords), len(y_coords), len(z_coords)),
        "dx": float(x_full[1] - x_full[0]) if len(x_full) > 1 else 1.0,
        "dy": float(y_full[1] - y_full[0]) if len(y_full) > 1 else 1.0,
        "dz": float(z_full[1] - z_full[0]) if len(z_full) > 1 else 1.0,
        "step": step,
        "time": time_val,
    }


def open_h5_for_parallel_read(filepath, comm):
    """Use MPI-enabled HDF5 reads when available."""
    if h5py.get_config().mpi:
        return h5py.File(filepath, "r", driver="mpio", comm=comm)
    return h5py.File(filepath, "r")


def parse_slice_spec(spec):
    """Parse one slice selector of the form axis:selector."""
    if ":" not in spec:
        raise ValueError(
            f"Invalid slice spec '{spec}'. Expected axis:selector, "
            "for example z:center, y:idx=10, x:frac=0.25, z:coord=0.5."
        )

    axis, selector = spec.split(":", 1)
    axis = axis.strip().lower()
    selector = selector.strip().lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Invalid slice axis '{axis}'.")

    if selector == "center":
        return axis, ("center", None)
    if selector.startswith("idx="):
        return axis, ("idx", int(selector.split("=", 1)[1]))
    if selector.startswith("frac="):
        return axis, ("frac", float(selector.split("=", 1)[1]))
    if selector.startswith("coord="):
        return axis, ("coord", float(selector.split("=", 1)[1]))

    raise ValueError(
        f"Unsupported slice selector '{selector}'. "
        "Use center, idx=<int>, frac=<0..1>, or coord=<float>."
    )


def resolve_slice_index(coords, selector):
    """Convert a parsed selector into a concrete plane index."""
    mode, value = selector
    npts = len(coords)
    if npts == 0:
        raise ValueError("Cannot select a slice from an empty axis.")

    if mode == "center":
        return npts // 2, "center"

    if mode == "idx":
        index = int(value)
        if index < 0 or index >= npts:
            raise ValueError(f"Slice index {index} is out of bounds for axis length {npts}.")
        return index, f"idx{index:05d}"

    if mode == "frac":
        frac = float(value)
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"Slice fraction {frac} is outside [0, 1].")
        index = int(np.rint(frac * (npts - 1)))
        return index, f"frac{frac:.6f}".replace(".", "p")

    if mode == "coord":
        coord = float(value)
        index = int(np.argmin(np.abs(coords - coord)))
        return index, f"coord{coord:.6f}".replace(".", "p")

    raise ValueError(f"Unknown selector mode '{mode}'.")


def gather_plane(axis, local_bounds, local_plane, shape, comm):
    """Gather one distributed 2D plane back to rank 0."""
    rank = comm.Get_rank()
    gathered = comm.gather((local_bounds, local_plane), root=0)
    if rank != 0:
        return None

    nx, ny, nz = shape
    if axis == "x":
        plane = np.zeros((ny, nz), dtype=np.float64)
        for bounds, piece in gathered:
            if piece is None:
                continue
            y0, y1, z0, z1 = bounds
            plane[y0:y1, z0:z1] = piece
        return plane

    if axis == "y":
        plane = np.zeros((nx, nz), dtype=np.float64)
        for bounds, piece in gathered:
            if piece is None:
                continue
            x0, x1, z0, z1 = bounds
            plane[x0:x1, z0:z1] = piece
        return plane

    plane = np.zeros((nx, ny), dtype=np.float64)
    for bounds, piece in gathered:
        if piece is None:
            continue
        x0, x1, y0, y1 = bounds
        plane[x0:x1, y0:y1] = piece
    return plane


def extract_plane_from_boxes_local(axis, local_box, local_field, plane_index):
    """Extract one rank-local piece of a plane from a generic 3D box decomposition."""
    sx, sy, sz = box_slices(local_box)

    bounds = None
    piece = None
    if axis == "x":
        if sx.start <= plane_index < sx.stop:
            local_index = plane_index - sx.start
            piece = np.asarray(local_field[local_index, :, :], dtype=np.float64)
            bounds = (sy.start, sy.stop, sz.start, sz.stop)
    elif axis == "y":
        if sy.start <= plane_index < sy.stop:
            local_index = plane_index - sy.start
            piece = np.asarray(local_field[:, local_index, :], dtype=np.float64)
            bounds = (sx.start, sx.stop, sz.start, sz.stop)
    else:
        if sz.start <= plane_index < sz.stop:
            local_index = plane_index - sz.start
            piece = np.asarray(local_field[:, :, local_index], dtype=np.float64)
            bounds = (sx.start, sx.stop, sy.start, sy.stop)

    return bounds, piece


def gather_plane_from_boxes(axis, local_box, local_field, shape, plane_index, comm):
    """Gather one plane from a generic 3D box decomposition."""
    local_bounds, local_plane = extract_plane_from_boxes_local(axis, local_box, local_field, plane_index)
    return gather_plane(axis, local_bounds, local_plane, shape, comm)


def extract_plane_parallel_local(filepath, dataset_name, axis, plane_index, meta, comm):
    """Read the rank-local piece of one requested HDF5 plane in parallel."""
    shape = meta["shape"]
    nx, ny, nz = shape
    x_ranges = split_axis(nx, comm.Get_size())
    x_start, x_stop = x_ranges[comm.Get_rank()]

    local_bounds = None
    local_plane = None
    with open_h5_for_parallel_read(filepath, comm) as hf:
        if dataset_name == "velocity_magnitude":
            vx_field = hf["fields"]["vx"]
            vy_field = hf["fields"]["vy"]
            vz_field = hf["fields"]["vz"]
            if axis == "x":
                if x_start <= plane_index < x_stop:
                    vx_plane = np.asarray(vx_field[plane_index, :ny, :nz], dtype=np.float64)
                    vy_plane = np.asarray(vy_field[plane_index, :ny, :nz], dtype=np.float64)
                    vz_plane = np.asarray(vz_field[plane_index, :ny, :nz], dtype=np.float64)
                    local_plane = np.sqrt(vx_plane**2 + vy_plane**2 + vz_plane**2)
                    local_bounds = (0, ny, 0, nz)
            elif axis == "y":
                vx_plane = np.asarray(vx_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                vy_plane = np.asarray(vy_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                vz_plane = np.asarray(vz_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                local_plane = np.sqrt(vx_plane**2 + vy_plane**2 + vz_plane**2)
                local_bounds = (x_start, x_stop, 0, nz)
            else:
                vx_plane = np.asarray(vx_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                vy_plane = np.asarray(vy_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                vz_plane = np.asarray(vz_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                local_plane = np.sqrt(vx_plane**2 + vy_plane**2 + vz_plane**2)
                local_bounds = (x_start, x_stop, 0, ny)
        else:
            field = hf["fields"][dataset_name]
            if axis == "x":
                if x_start <= plane_index < x_stop:
                    local_plane = np.asarray(field[plane_index, :ny, :nz], dtype=np.float64)
                    local_bounds = (0, ny, 0, nz)
            elif axis == "y":
                local_plane = np.asarray(field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                local_bounds = (x_start, x_stop, 0, nz)
            else:
                local_plane = np.asarray(field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                local_bounds = (x_start, x_stop, 0, ny)

    return local_bounds, local_plane


def extract_plane_parallel(filepath, dataset_name, axis, plane_index, meta, comm):
    """Read only the requested HDF5 plane in parallel."""
    local_bounds, local_plane = extract_plane_parallel_local(filepath, dataset_name, axis, plane_index, meta, comm)
    return gather_plane(axis, local_bounds, local_plane, meta["shape"], comm)


def compute_local_vorticity_fields(filepath, meta, comm, backend_name):
    """Compute distributed real-space vorticity components with HeFFTe."""
    shape = meta["shape"]
    proc_grid = choose_proc_grid(shape, comm.Get_size())
    boxes = build_boxes(shape, proc_grid)
    local_box = boxes[comm.Get_rank()]
    local_shape = box_shape(local_box)
    local_vx, local_vy, local_vz = read_structured_local_fields(filepath, local_box, comm)

    backend = get_backend(backend_name)
    plan = heffte.fft3d(backend, local_box, local_box, comm)
    KX, KY, KZ = local_wavenumber_mesh(shape, local_box, meta["dx"], meta["dy"], meta["dz"])

    vx_k = forward_field(plan, local_vx).reshape(local_shape, order="C")
    vy_k = forward_field(plan, local_vy).reshape(local_shape, order="C")
    vz_k = forward_field(plan, local_vz).reshape(local_shape, order="C")

    omega_x_k = 1j * (KY * vz_k - KZ * vy_k)
    omega_y_k = 1j * (KZ * vx_k - KX * vz_k)
    omega_z_k = 1j * (KX * vy_k - KY * vx_k)

    omega_x = backward_field(plan, omega_x_k, local_shape)
    omega_y = backward_field(plan, omega_y_k, local_shape)
    omega_z = backward_field(plan, omega_z_k, local_shape)

    return {
        "local_box": local_box,
        "omega_x": omega_x,
        "omega_y": omega_y,
        "omega_z": omega_z,
        "vorticity_magnitude": np.sqrt(omega_x**2 + omega_y**2 + omega_z**2),
    }


def output_stem(source_path, field_label):
    """Build a clean default filename stem with the plotted field name once in front."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    marker = "_sampled_data"
    if marker in base:
        _, suffix = base.split(marker, 1)
        return f"{field_label}{marker}{suffix}"
    if base == field_label or base.startswith(f"{field_label}_"):
        return base
    if base.endswith(f"_{field_label}"):
        return base[: -(len(field_label) + 1)]
    return f"{field_label}_{base}"


def output_name(data_file, field_label, axis, slice_tag, output_format):
    """Return the default output name for one slice image."""
    directory = os.path.dirname(os.path.abspath(data_file))
    output_dir = os.path.join(directory, "slice_plots")
    os.makedirs(output_dir, exist_ok=True)
    base = output_stem(data_file, field_label)
    if slice_tag in {"xy_center", "xy_face", "yz_face", "zx_face"}:
        return os.path.join(output_dir, f"{base}_{slice_tag}.{output_format}")
    return os.path.join(output_dir, f"{base}_{PLANE_NAMES[axis]}_{slice_tag}.{output_format}")


def output_source_path(data_file, dataset_name, field_family):
    """Return the path whose stem should drive default output naming for one field."""
    if field_family != "scalar":
        return data_file

    with h5py.File(data_file, "r") as hf:
        dataset = hf["fields"][dataset_name]
        source_path = dataset.attrs.get("source_path")
        source_h5 = dataset.attrs.get("source_h5")
        source_txt = dataset.attrs.get("source_txt")

    for candidate in (source_path, source_h5, source_txt):
        if candidate:
            return str(candidate)
    return data_file


def axis_bounds(coords):
    """Return lower edge, upper edge, and spacing for one axis coordinate array."""
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) == 1:
        delta = 1.0
    else:
        delta = float(coords[1] - coords[0])
    return float(coords[0]), float(coords[-1] + delta), delta


def build_yt_slice_dataset(plane, meta, axis, plane_index, field_label):
    """Construct a one-cell-thick yt dataset from one extracted 2D plane."""
    import yt

    plane = np.asarray(plane, dtype=np.float64)
    x0, x1, dx = axis_bounds(meta["x"])
    y0, y1, dy = axis_bounds(meta["y"])
    z0, z1, dz = axis_bounds(meta["z"])
    plane_coord = float(meta[axis][plane_index])

    if axis == "x":
        data = plane[np.newaxis, :, :]
        bbox = np.array([[plane_coord, plane_coord + dx], [y0, y1], [z0, z1]], dtype=np.float64)
    elif axis == "y":
        data = plane[:, np.newaxis, :]
        bbox = np.array([[x0, x1], [plane_coord, plane_coord + dy], [z0, z1]], dtype=np.float64)
    else:
        data = plane[:, :, np.newaxis]
        bbox = np.array([[x0, x1], [y0, y1], [plane_coord, plane_coord + dz]], dtype=np.float64)

    dataset = yt.load_uniform_grid(
        {field_label: data},
        data.shape,
        bbox=bbox,
        nprocs=1,
        periodicity=(False, False, False),
        unit_system="cgs",
    )
    return dataset, ("stream", field_label)


def render_plane_image(
    plane,
    meta,
    axis,
    plane_index,
    field_label,
    latex_label,
    cmap,
    width,
    output,
    plot,
    save_dpi,
    figure_size,
):
    """Render one plane to disk on rank 0 with yt SlicePlot."""
    import yt

    plane = np.asarray(plane, dtype=np.float64).copy()
    plane[np.abs(plane) < 1.0e-12] = 0.0
    plane = np.round(plane, decimals=10)
    zmin = float(np.min(plane))
    zmax = float(np.max(plane))
    dataset, yt_field = build_yt_slice_dataset(plane, meta, axis, plane_index, field_label)
    slice_plot = yt.SlicePlot(dataset, axis, yt_field, center="c", origin="native")
    slice_plot.set_log(yt_field, False)
    slice_plot.set_cmap(yt_field, cmap)
    slice_plot.set_axes_unit("unitary")
    slice_plot.set_font(_yt_font_style())
    slice_plot.set_figure_size(float(figure_size))
    data_res = max(plane.shape)
    slice_plot.set_buff_size((data_res, data_res))
    if not np.isclose(zmin, zmax):
        slice_plot.set_zlim(yt_field, zmin=zmin, zmax=zmax)
    slice_plot.set_minorticks("all", False)
    slice_plot.set_colorbar_minorticks("all", False)
    if width is not None:
        slice_plot.set_width((float(width), "code_length"))

    slice_plot.render()
    window_plot = slice_plot.plots[yt_field]
    window_plot.cb.set_label(latex_label)
    image = window_plot.axes.images[0]
    image.set_interpolation("bicubic")
    if hasattr(image, "set_interpolation_stage"):
        image.set_interpolation_stage("rgba")
    saved_paths = [window_plot.save(output, mpl_kwargs={"dpi": save_dpi})]
    for saved_path in saved_paths:
        print(f"Saved: {saved_path}")
    if plot:
        slice_plot.show()
    return list(saved_paths)


# Legacy matplotlib renderer reference kept for future use.
# import matplotlib.pyplot as plt
#
# def render_plane_image_matplotlib(
#     plane,
#     meta,
#     axis,
#     plane_index,
#     field_label,
#     latex_label,
#     cmap,
#     width,
#     output,
#     plot,
#     save_dpi,
#     figure_size,
# ):
#     plane = np.asarray(plane, dtype=np.float64).copy()
#     plane[np.abs(plane) < 1.0e-12] = 0.0
#     plane = np.round(plane, decimals=10)
#     info = plane_axes_and_extent(meta, axis)
#     with plt.rc_context(_mpl_plot_style()):
#         fig, ax = plt.subplots(figsize=(figure_size, figure_size))
#         image = ax.imshow(
#             plane,
#             origin="lower",
#             extent=info["extent"],
#             cmap=cmap,
#             interpolation="bicubic",
#             aspect="equal",
#         )
#         ax.set_box_aspect(1)
#         ax.set_xlabel(rf"${info['horizontal_name']}$", fontsize=24)
#         ax.set_ylabel(rf"${info['vertical_name']}$", fontsize=24)
#         apply_width(ax, meta, axis, width)
#         colorbar = fig.colorbar(image, ax=ax, label=latex_label)
#         fig.tight_layout()
#         fig.savefig(output, dpi=save_dpi)
#         if plot:
#             plt.show()
#         plt.close(fig)


def build_slice_requests(meta, slice_specs, default_axis):
    """Resolve one or more user slice specs into concrete requests."""
    requests = []
    if slice_specs:
        raw_specs = list(slice_specs)
        for spec in raw_specs:
            axis, selector = parse_slice_spec(spec)
            coords = meta[axis]
            plane_index, slice_tag = resolve_slice_index(coords, selector)
            requests.append((axis, plane_index, slice_tag))
        return requests

    default_requests = [
        ("z", ("center", None), "xy_center"),
        ("z", ("idx", 0), "xy_face"),
        ("x", ("idx", 0), "yz_face"),
        ("y", ("idx", 0), "zx_face"),
    ]
    seen = set()
    for axis, selector, default_tag in default_requests:
        coords = meta[axis]
        plane_index, _ = resolve_slice_index(coords, selector)
        key = (axis, plane_index)
        if key in seen:
            continue
        seen.add(key)
        requests.append((axis, plane_index, default_tag))
    return requests


def resolve_requested_fields(filepath, field_names):
    """Resolve requested field names against the available HDF5-backed field specs."""
    field_lookup = available_field_specs(filepath)
    if field_names:
        requested_fields = list(field_names)
    else:
        scalar_fields = [name for name, spec in field_lookup.items() if spec[3] == "scalar"]
        requested_fields = ["velocity_magnitude", "vorticity_magnitude", *scalar_fields]

    missing = [name for name in requested_fields if name not in field_lookup]
    if missing:
        available = ", ".join(sorted(field_lookup))
        raise ValueError(f"Unsupported field(s): {', '.join(missing)}. Available fields: {available}")

    return [field_lookup[name] for name in requested_fields]


def run_visualization(
    data_file,
    axis="z",
    field_names=None,
    cmap="RdBu_r",
    width=None,
    output=None,
    plot=False,
    comm=None,
    slice_specs=None,
    assume_structured_h5=False,
    backend_name="heffte_fftw",
    output_format="pdf",
    save_dpi=300,
    figure_size=8.0,
    save_slice_data=True,
    slice_data_output=None,
):
    """Render one or more slice images from a structured HDF5 velocity file."""
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    prepared_path = data_file if assume_structured_h5 else ensure_structured_h5(data_file)
    field_specs = resolve_requested_fields(prepared_path, field_names)
    meta = read_grid_metadata(prepared_path)
    requests = build_slice_requests(meta, slice_specs, axis)
    saved_slice_data_path = slice_data_output or default_slice_data_output(prepared_path)

    if output and (len(requests) != 1 or len(field_specs) != 1):
        raise ValueError("--output can only be used when rendering a single slice.")

    needs_vorticity = any(spec[3] == "vorticity" for spec in field_specs)
    vorticity_cache = None

    if rank == 0:
        print()
        print("-" * 60, flush=True)
        print("SLICE RENDERING", flush=True)
        print("-" * 60, flush=True)
        print(f"Loading {prepared_path} with {comm.Get_size()} MPI ranks...", flush=True)
        print(f"Structured domain dimensions: {meta['shape']}", flush=True)
        print(f"Rendering {len(requests)} slice(s) for {len(field_specs)} field set(s)...", flush=True)
        if save_slice_data:
            print(f"Saving reusable slice data to: {saved_slice_data_path}", flush=True)

    if save_slice_data:
        init_start = time.perf_counter()
        if rank == 0:
            initialize_slice_data_file(
                saved_slice_data_path,
                meta,
                field_specs,
                requests,
                data_file,
                prepared_path,
                backend_name,
            )
        comm.Barrier()
        log_rank0(rank, f"Initialized slice-data file in {time.perf_counter() - init_start:.2f}s")

    if needs_vorticity:
        vorticity_start = time.perf_counter()
        log_rank0(rank, "Computing distributed vorticity field with HeFFTe inverse FFT...")
        vorticity_cache = compute_local_vorticity_fields(prepared_path, meta, comm, backend_name)
        comm.Barrier()
        log_rank0(rank, f"Completed distributed vorticity field in {time.perf_counter() - vorticity_start:.2f}s")

    outputs = []
    # DANE can hang in the legacy MPI slice-data write path for x-normal slices,
    # where only one rank owns the requested plane. Keep the old branch below so
    # it can be restored easily if we ever want to revisit distributed slice writes.
    use_legacy_parallel_slice_data_write = False
    for dataset_name, field_label, latex_label, field_family in field_specs:
        log_rank0(rank, f"Field: {field_label}")
        for axis_name, plane_index, slice_tag in requests:
            slice_start = time.perf_counter()
            stored_slice_tag = storage_slice_tag(axis_name, slice_tag)
            if field_family == "vorticity":
                local_bounds, local_plane = extract_plane_from_boxes_local(
                    axis_name,
                    vorticity_cache["local_box"],
                    vorticity_cache[dataset_name],
                    plane_index,
                )
            else:
                local_bounds, local_plane = extract_plane_parallel_local(
                    prepared_path,
                    dataset_name,
                    axis_name,
                    plane_index,
                    meta,
                    comm,
                )

            gather_start = time.perf_counter()
            plane = gather_plane(axis_name, local_bounds, local_plane, meta["shape"], comm)
            log_rank0(
                rank,
                f"  Gather finished for {field_label}/{stored_slice_tag} in "
                f"{time.perf_counter() - gather_start:.2f}s",
            )
            if save_slice_data:
                write_start = time.perf_counter()
                if use_legacy_parallel_slice_data_write and h5py.get_config().mpi:
                    # Legacy MPI-HDF5 slice writes retained for reference. This path
                    # hung on DANE for x-normal slices when only one rank had data.
                    write_slice_plane_parallel(
                        saved_slice_data_path,
                        field_label,
                        stored_slice_tag,
                        axis_name,
                        local_bounds,
                        local_plane,
                        comm,
                    )
                    comm.Barrier()
                else:
                    # Gathered planes are already needed for rank-0 rendering, so
                    # write the reusable slice-data file serially on rank 0.
                    if rank == 0:
                        write_slice_plane_serial(saved_slice_data_path, field_label, stored_slice_tag, plane)
                    comm.Barrier()
                log_rank0(
                    rank,
                    f"  Slice-data write finished for {field_label}/{stored_slice_tag} in "
                    f"{time.perf_counter() - write_start:.2f}s",
                )

            rendered_paths = []
            if rank == 0:
                coord_value = meta[axis_name][plane_index]
                print(
                    f"  Slice normal={axis_name}, index={plane_index}, "
                    f"coord={coord_value:.6g}, step={meta['step']}, time={meta['time']:.6g}"
                ,
                    flush=True,
                )
                rendered = output or output_name(
                    output_source_path(prepared_path, dataset_name, field_family),
                    field_label,
                    axis_name,
                    slice_tag,
                    output_format,
                )
                render_start = time.perf_counter()
                rendered_paths = render_plane_image(
                    plane,
                    meta,
                    axis_name,
                    plane_index,
                    field_label,
                    latex_label,
                    cmap,
                    width,
                    rendered,
                    plot,
                    save_dpi,
                    figure_size,
                )
                print(
                    f"  Render finished for {field_label}/{stored_slice_tag} in "
                    f"{time.perf_counter() - render_start:.2f}s",
                    flush=True,
                )
                outputs.extend(rendered_paths)
            rendered_paths = comm.bcast(rendered_paths, root=0)
            comm.Barrier()
            log_rank0(
                rank,
                f"  Completed slice {field_label}/{stored_slice_tag} in "
                f"{time.perf_counter() - slice_start:.2f}s",
            )
            if rank != 0:
                outputs.extend(rendered_paths)

    return outputs, (saved_slice_data_path if save_slice_data else None)


def main():
    parser = argparse.ArgumentParser(description="Render one or more velocity slices from TXT or structured HDF5")
    parser.add_argument("data_file", help="Path to a SampledData .txt or structured .h5 file")
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        help="Slice spec axis:selector, e.g. z:center, y:idx=10, x:frac=0.25, z:coord=0.5",
    )
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"], help="Default center-slice axis if --slice is omitted")
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        help="Field to plot. Repeat to render multiple fields. If omitted, render velocity, vorticity, and any appended scalar fields.",
    )
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap")
    parser.add_argument("--width", type=float, default=None, help="Optional square plot width in domain units")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"], help="Output image format. Default is pdf.")
    parser.add_argument("--dpi", type=int, default=600, help="Raster save DPI. Default is 600.")
    parser.add_argument("--figsize", type=float, default=8.0, help="Square figure size in inches. Default is 8.0.")
    parser.add_argument("--output", default=None, help="Optional output path for a single slice")
    parser.add_argument("--plot", action="store_true", help="Also display the plot on rank 0 after saving")
    parser.add_argument("--no-slice-data", action="store_true", help="Skip writing the combined *_slices.h5 file.")
    parser.add_argument("--slice-data-output", default=None, help="Optional path for the combined slice-data HDF5 file.")
    parser.add_argument(
        "--backend",
        default="heffte_fftw",
        choices=["heffte_fftw", "heffte_stock"],
        help="HeFFTe backend used when deriving vorticity from FFTs.",
    )
    args = parser.parse_args()
    run_visualization(
        args.data_file,
        axis=args.axis,
        field_names=args.field,
        cmap=args.cmap,
        width=args.width,
        output=args.output,
        plot=args.plot,
        slice_specs=args.slice,
        backend_name=args.backend,
        output_format=args.format,
        save_dpi=args.dpi,
        figure_size=args.figsize,
        save_slice_data=not args.no_slice_data,
        slice_data_output=args.slice_data_output,
    )
