"""Scalable slice-plot workflow for structured velocity HDF5 files."""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from postprocess_lib.prepare import ensure_structured_h5


FIELD_MAP = {
    "vmag": ("velocity_magnitude", "velocity_magnitude", r"$|\mathbf{u}|$"),
    "mag": ("velocity_magnitude", "velocity_magnitude", r"$|\mathbf{u}|$"),
    "velocity_magnitude": ("velocity_magnitude", "velocity_magnitude", r"$|\mathbf{u}|$"),
    "vx": ("vx", "velocity_x", r"$u_1$"),
    "vy": ("vy", "velocity_y", r"$u_2$"),
    "vz": ("vz", "velocity_z", r"$u_3$"),
    "velocity_x": ("vx", "velocity_x", r"$u_1$"),
    "velocity_y": ("vy", "velocity_y", r"$u_2$"),
    "velocity_z": ("vz", "velocity_z", r"$u_3$"),
}

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
    if field_name not in FIELD_MAP:
        raise ValueError(f"Unsupported field '{field_name}'.")
    return FIELD_MAP[field_name]


def _plot_style():
    return {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 20,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
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


def gather_plane(axis, plane_index, local_bounds, local_plane, shape, comm):
    """Gather one distributed 2D plane back to rank 0."""
    rank = comm.Get_rank()
    gathered = comm.gather((local_bounds, local_plane), root=0)
    if rank != 0:
        return None

    nx, ny, nz = shape
    if axis == "x":
        for (x_start, x_stop), plane in gathered:
            if plane is not None:
                return plane
        raise ValueError(f"Unable to assemble x-slice at index {plane_index}.")

    if axis == "y":
        global_plane = np.zeros((nx, nz), dtype=np.float64)
        for (x_start, x_stop), plane in gathered:
            if plane is not None and x_stop > x_start:
                global_plane[x_start:x_stop, :] = plane
        return global_plane

    global_plane = np.zeros((nx, ny), dtype=np.float64)
    for (x_start, x_stop), plane in gathered:
        if plane is not None and x_stop > x_start:
            global_plane[x_start:x_stop, :] = plane
    return global_plane


def extract_plane_parallel(filepath, dataset_name, axis, plane_index, meta, comm):
    """Read only the requested HDF5 plane in parallel."""
    shape = meta["shape"]
    nx, ny, nz = shape
    x_ranges = split_axis(nx, comm.Get_size())
    x_start, x_stop = x_ranges[comm.Get_rank()]

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
            elif axis == "y":
                vx_plane = np.asarray(vx_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                vy_plane = np.asarray(vy_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                vz_plane = np.asarray(vz_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                local_plane = np.sqrt(vx_plane**2 + vy_plane**2 + vz_plane**2)
            else:
                vx_plane = np.asarray(vx_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                vy_plane = np.asarray(vy_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                vz_plane = np.asarray(vz_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                local_plane = np.sqrt(vx_plane**2 + vy_plane**2 + vz_plane**2)
        else:
            field = hf["fields"][dataset_name]
            if axis == "x":
                if x_start <= plane_index < x_stop:
                    local_plane = np.asarray(field[plane_index, :ny, :nz], dtype=np.float64)
            elif axis == "y":
                local_plane = np.asarray(field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
            else:
                local_plane = np.asarray(field[x_start:x_stop, :ny, plane_index], dtype=np.float64)

    return gather_plane(axis, plane_index, (x_start, x_stop), local_plane, shape, comm)


def axis_extent(coords):
    """Return imshow extent bounds from 1D coordinates."""
    if len(coords) == 1:
        delta = 1.0
    else:
        delta = float(coords[1] - coords[0])
    return float(coords[0]), float(coords[-1] + delta)


def plane_axes_and_extent(meta, axis):
    """Return plotting metadata for one plane orientation."""
    x_coords = meta["x"]
    y_coords = meta["y"]
    z_coords = meta["z"]
    if axis == "x":
        horizontal = z_coords
        vertical = y_coords
    elif axis == "y":
        horizontal = z_coords
        vertical = x_coords
    else:
        horizontal = y_coords
        vertical = x_coords

    xmin, xmax = axis_extent(horizontal)
    ymin, ymax = axis_extent(vertical)
    return {
        "horizontal_name": PLANE_AXES[axis][0],
        "vertical_name": PLANE_AXES[axis][1],
        "extent": [xmin, xmax, ymin, ymax],
    }


def output_name(data_file, field_label, axis, slice_tag):
    """Return the default output name for one slice image."""
    directory = os.path.dirname(os.path.abspath(data_file))
    output_dir = os.path.join(directory, "slice_plots")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(data_file))[0]
    if slice_tag in {"xy_center", "xy_face", "yz_face", "zx_face"}:
        return os.path.join(output_dir, f"{base}_{slice_tag}_{field_label}.png")
    return os.path.join(output_dir, f"{base}_{PLANE_NAMES[axis]}_{slice_tag}_{field_label}.png")


def apply_width(ax, meta, axis, width):
    """Zoom the plot around the plane center when width is provided."""
    if width is None:
        return

    axis_info = plane_axes_and_extent(meta, axis)
    x0, x1, y0, y1 = axis_info["extent"]
    xmid = 0.5 * (x0 + x1)
    ymid = 0.5 * (y0 + y1)
    half = 0.5 * float(width)
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)


def render_plane_image(plane, meta, axis, plane_index, field_label, latex_label, cmap, width, output, plot):
    """Render one plane to PNG on rank 0."""
    info = plane_axes_and_extent(meta, axis)
    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        image = ax.imshow(
            plane,
            origin="lower",
            extent=info["extent"],
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )
        axis_label_map = {"x": r"$x$", "y": r"$y$", "z": r"$z$"}
        ax.set_xlabel(axis_label_map[info["horizontal_name"]], fontsize=24)
        ax.set_ylabel(axis_label_map[info["vertical_name"]], fontsize=24)
        apply_width(ax, meta, axis, width)
        tick_values = np.linspace(float(np.min(plane)), float(np.max(plane)), 8)
        if np.allclose(tick_values[0], tick_values[-1]):
            tick_values = np.array([tick_values[0]])
        colorbar = fig.colorbar(image, ax=ax, label=latex_label, ticks=tick_values)
        colorbar.ax.tick_params(labelsize=20)
        colorbar.ax.set_yticklabels([f"{value:.2g}" for value in tick_values])
        colorbar.set_label(latex_label, size=24)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        print(f"Saved: {output}")
        if plot:
            plt.show()
        plt.close(fig)


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


def run_visualization(
    data_file,
    axis="z",
    field_name="vx",
    cmap="RdBu_r",
    width=None,
    output=None,
    plot=False,
    comm=None,
    slice_specs=None,
    assume_structured_h5=False,
):
    """Render one or more slice images from a structured HDF5 velocity file."""
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    prepared_path = data_file if assume_structured_h5 else ensure_structured_h5(data_file)
    dataset_name, field_label, latex_label = canonical_field_name(field_name)
    meta = read_grid_metadata(prepared_path)
    requests = build_slice_requests(meta, slice_specs, axis)

    if output and len(requests) != 1:
        raise ValueError("--output can only be used when rendering a single slice.")

    if rank == 0:
        print()
        print("-" * 60)
        print("SLICE RENDERING")
        print("-" * 60)
        print(f"Loading {prepared_path} with {comm.Get_size()} MPI ranks...")
        print(f"Structured domain dimensions: {meta['shape']}")
        print(f"Rendering {len(requests)} slice(s) for field '{field_label}'...")

    outputs = []
    for axis_name, plane_index, slice_tag in requests:
        plane = extract_plane_parallel(prepared_path, dataset_name, axis_name, plane_index, meta, comm)
        rendered = None
        if rank == 0:
            coord_value = meta[axis_name][plane_index]
            print(
                f"  Slice normal={axis_name}, index={plane_index}, "
                f"coord={coord_value:.6g}, step={meta['step']}, time={meta['time']:.6g}"
            )
            rendered = output or output_name(prepared_path, field_label, axis_name, slice_tag)
            render_plane_image(plane, meta, axis_name, plane_index, field_label, latex_label, cmap, width, rendered, plot)
            outputs.append(rendered)
        rendered = comm.bcast(rendered, root=0)
        comm.Barrier()
        if rank != 0:
            outputs.append(rendered)

    return outputs


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
        default="velocity_magnitude",
        choices=sorted(FIELD_MAP),
        help="Field to plot. Default is velocity magnitude.",
    )
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap")
    parser.add_argument("--width", type=float, default=None, help="Optional square plot width in domain units")
    parser.add_argument("--output", default=None, help="Optional output PNG path for a single slice")
    parser.add_argument("--plot", action="store_true", help="Also display the plot on rank 0 after saving")
    args = parser.parse_args()
    run_visualization(
        args.data_file,
        axis=args.axis,
        field_name=args.field,
        cmap=args.cmap,
        width=args.width,
        output=args.output,
        plot=args.plot,
        slice_specs=args.slice,
    )
