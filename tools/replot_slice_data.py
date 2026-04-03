#!/usr/bin/env python3
"""Inspect and replot saved 2D slice data from a combined *_slices.h5 file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocess_vis.slice_data import list_available_slices
from postprocess_vis.slice_data import load_saved_slice
from postprocess_vis.slice_data import plane_extent_from_arrays


def _yt_font_style():
    return {
        "family": "serif",
        "size": 18,
    }


def print_summary(filepath):
    """Print a concise summary of what can be replotted from one slice-data file."""
    summary = list_available_slices(filepath)
    print(f"Slice file: {os.path.abspath(filepath)}")
    print(f"Source file: {summary['source_file']}")
    print(f"Source HDF5: {summary['source_h5']}")
    print(f"Step: {summary['step']}")
    print(f"Time: {summary['time']:.6g}")
    print(f"Grid shape: {summary['grid_shape']}")
    print("Available fields and slices:")
    for field_name, slice_tags in summary["fields"].items():
        print(f"  {field_name}: {', '.join(slice_tags)}")


def default_output_name(slice_file, field_name, slice_tag, output_format):
    """Return the default output path for one replotted saved slice."""
    directory = os.path.dirname(os.path.abspath(slice_file))
    output_dir = os.path.join(directory, "slice_replots")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(slice_file))[0]
    return os.path.join(output_dir, f"{base}_{slice_tag}_{field_name}.{output_format}")


def build_yt_slice_dataset(slice_file, field_name, slice_tag):
    """Construct a one-cell-thick yt uniform-grid dataset from one saved slice."""
    import yt

    saved = load_saved_slice(slice_file, field_name, slice_tag)
    attrs = saved["attrs"]
    axis = str(attrs["axis"])
    plane_coord = float(attrs["plane_coord"])
    values = np.asarray(saved["values"], dtype=np.float64)

    with h5py.File(slice_file, "r") as hf:
        x_coords = np.asarray(hf["grid"]["x"][:], dtype=np.float64)
        y_coords = np.asarray(hf["grid"]["y"][:], dtype=np.float64)
        z_coords = np.asarray(hf["grid"]["z"][:], dtype=np.float64)

    def bounds(coords):
        if len(coords) == 1:
            delta = 1.0
        else:
            delta = float(coords[1] - coords[0])
        return float(coords[0]), float(coords[-1] + delta), delta

    x0, x1, dx = bounds(x_coords)
    y0, y1, dy = bounds(y_coords)
    z0, z1, dz = bounds(z_coords)

    if axis == "x":
        data = values[np.newaxis, :, :]
        bbox = np.array([[plane_coord, plane_coord + dx], [y0, y1], [z0, z1]], dtype=np.float64)
    elif axis == "y":
        data = values[:, np.newaxis, :]
        bbox = np.array([[x0, x1], [plane_coord, plane_coord + dy], [z0, z1]], dtype=np.float64)
    else:
        data = values[:, :, np.newaxis]
        bbox = np.array([[x0, x1], [y0, y1], [plane_coord, plane_coord + dz]], dtype=np.float64)

    dataset = yt.load_uniform_grid(
        {field_name: data},
        data.shape,
        bbox=bbox,
        nprocs=1,
        periodicity=(False, False, False),
        unit_system="cgs",
    )
    return dataset, ("stream", field_name), saved


def render_saved_slice(
    slice_file,
    field_name,
    slice_tag,
    cmap,
    width,
    vmin,
    vmax,
    output,
    output_format,
    plot,
    dpi,
    figure_size,
):
    """Render one saved slice plane to disk and/or screen with yt bicubic output."""
    import yt

    dataset, yt_field, saved = build_yt_slice_dataset(slice_file, field_name, slice_tag)
    values = np.asarray(saved["values"], dtype=np.float64).copy()
    values[np.abs(values) < 1.0e-12] = 0.0
    values = np.round(values, decimals=10)

    attrs = saved["attrs"]
    output_path = output or default_output_name(slice_file, field_name, slice_tag, output_format)
    zmin = float(np.min(values) if vmin is None else vmin)
    zmax = float(np.max(values) if vmax is None else vmax)

    slice_plot = yt.SlicePlot(dataset, str(attrs["axis"]), yt_field, center="c", origin="native")
    slice_plot.set_log(yt_field, False)
    slice_plot.set_cmap(yt_field, cmap)
    slice_plot.set_axes_unit("unitary")
    slice_plot.set_font(_yt_font_style())
    slice_plot.set_figure_size(float(figure_size))
    data_res = max(values.shape)
    slice_plot.set_buff_size((data_res, data_res))
    if not np.isclose(zmin, zmax):
        slice_plot.set_zlim(yt_field, zmin=zmin, zmax=zmax)
    slice_plot.set_minorticks("all", False)
    slice_plot.set_colorbar_minorticks("all", False)
    if width is not None:
        slice_plot.set_width((float(width), "code_length"))

    slice_plot.render()
    window_plot = slice_plot.plots[yt_field]
    window_plot.cb.set_label(str(attrs["plot_label"]))
    image = window_plot.axes.images[0]
    image.set_interpolation("bicubic")
    if hasattr(image, "set_interpolation_stage"):
        image.set_interpolation_stage("rgba")
    saved_path = window_plot.save(output_path, mpl_kwargs={"dpi": dpi})
    print(f"Saved: {saved_path}")
    if plot:
        slice_plot.show()


# Legacy matplotlib replot path kept for future reference.
# import matplotlib.pyplot as plt
#
# def render_saved_slice_matplotlib(
#     slice_file,
#     field_name,
#     slice_tag,
#     cmap,
#     width,
#     vmin,
#     vmax,
#     output,
#     output_format,
#     plot,
#     dpi,
#     figure_size,
# ):
#     saved = load_saved_slice(slice_file, field_name, slice_tag)
#     values = np.asarray(saved["values"], dtype=np.float64).copy()
#     values[np.abs(values) < 1.0e-12] = 0.0
#     values = np.round(values, decimals=10)
#     horizontal_coords = saved["coord_horizontal"]
#     vertical_coords = saved["coord_vertical"]
#     extent = plane_extent_from_arrays(horizontal_coords, vertical_coords)
#     output_path = output or default_output_name(slice_file, field_name, slice_tag, output_format)
#     fig, ax = plt.subplots(figsize=(figure_size, figure_size))
#     image = ax.imshow(
#         values,
#         origin="lower",
#         extent=extent,
#         cmap=cmap,
#         interpolation="bicubic",
#         aspect="equal",
#         vmin=vmin,
#         vmax=vmax,
#     )
#     fig.colorbar(image, ax=ax, label=str(saved["attrs"]["plot_label"]))
#     fig.tight_layout()
#     fig.savefig(output_path, dpi=dpi)
#     if plot:
#         plt.show()
#     plt.close(fig)


def print_saved_slice_metadata(slice_file, field_name, slice_tag):
    """Print metadata for one selected saved slice."""
    saved = load_saved_slice(slice_file, field_name, slice_tag)
    attrs = saved["attrs"]
    print(f"Selected field: {field_name}")
    print(f"Selected slice: {slice_tag}")
    print(f"Axis: {attrs['axis']}")
    print(f"Plane index: {int(attrs['plane_index'])}")
    print(f"Plane coordinate: {float(attrs['plane_coord']):.6g}")
    print(f"Horizontal axis: {attrs['horizontal_axis']}")
    print(f"Vertical axis: {attrs['vertical_axis']}")
    print(f"Values shape: {saved['values'].shape}")


def print_yt_summary(slice_file, field_name, slice_tag):
    """Construct a one-cell-thick yt uniform-grid dataset and print its summary."""
    dataset, _, _ = build_yt_slice_dataset(slice_file, field_name, slice_tag)

    print("Constructed yt dataset:")
    print(f"  Dimensions: {tuple(int(v) for v in dataset.domain_dimensions)}")
    print(f"  Left edge: {tuple(float(v) for v in dataset.domain_left_edge)}")
    print(f"  Right edge: {tuple(float(v) for v in dataset.domain_right_edge)}")
    print(f"  Available yt fields: {dataset.field_list}")


def main():
    parser = argparse.ArgumentParser(description="Inspect and replot saved 2D slice data from *_slices.h5 files.")
    parser.add_argument("slice_file", help="Path to the combined *_slices.h5 file")
    parser.add_argument("--list", action="store_true", help="Print the available fields and slices, then exit.")
    parser.add_argument("--field", default=None, help="Field name to replot, for example velocity_magnitude.")
    parser.add_argument("--slice", dest="slice_tag", default=None, help="Slice tag to replot, for example xy_center.")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap")
    parser.add_argument("--width", type=float, default=None, help="Optional square plot width in domain units")
    parser.add_argument("--vmin", type=float, default=None, help="Optional lower colorbar limit")
    parser.add_argument("--vmax", type=float, default=None, help="Optional upper colorbar limit")
    parser.add_argument("--output", default=None, help="Optional output image path")
    parser.add_argument("--format", default="pdf", choices=["png", "pdf"], help="Default output format when --output is omitted.")
    parser.add_argument("--plot", action="store_true", help="Also display the plot after saving")
    parser.add_argument("--dpi", type=int, default=600, help="Raster save DPI. Default is 600.")
    parser.add_argument("--figsize", type=float, default=8.0, help="Square figure size in inches. Default is 8.0.")
    parser.add_argument(
        "--yt-info",
        action="store_true",
        help="Construct a one-cell-thick yt uniform-grid dataset for the selected slice and print its summary.",
    )
    args = parser.parse_args()

    if args.list:
        print_summary(args.slice_file)
        return
    if args.field is None and args.slice_tag is None:
        print_summary(args.slice_file)
        return
    if args.field is None or args.slice_tag is None:
        raise SystemExit("--field and --slice must be provided together.")

    print_saved_slice_metadata(args.slice_file, args.field, args.slice_tag)
    if args.yt_info:
        print_yt_summary(args.slice_file, args.field, args.slice_tag)
    render_saved_slice(
        args.slice_file,
        args.field,
        args.slice_tag,
        args.cmap,
        args.width,
        args.vmin,
        args.vmax,
        args.output,
        args.format,
        args.plot,
        args.dpi,
        args.figsize,
    )


if __name__ == "__main__":
    main()
