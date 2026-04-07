#!/usr/bin/env python3
"""Inspect and replot saved 2D slice data from a combined *_slices.h5 file.

Example commands:
  Notes:
    --field selects the saved variable to replot.
    --slice selects which saved plane of that variable to render.

  List available fields and slices:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --list

  Render one yt slice image:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center

  Render a separate contour plot with the default yt backend:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --contour

  Render contours with a fixed number of levels:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --contour --contour-levels 20

  Render contours at explicit values:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --contour --contour-values 0.5,1.0,2.0,4.0,8.0

  Render normalized vorticity and contour values in normalized units:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --normalize vorticity --contour --contour-values 0.5,1.0,2.0

  Batch-process multiple slice files and save into each file's own slice_replots folder:
    python tools/replot_slice_data.py data/run1/SampledData0_slices.h5 data/run2/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --contour

  Compare overlaid contours from two or three slice files:
    python tools/replot_slice_data.py data/run1/SampledData0_slices.h5 data/run2/SampledData0_slices.h5 data/run3/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --contour --compare

  Compare two slice files using fixed contour values:
    python tools/replot_slice_data.py data/run1/SampledData0_slices.h5 data/run2/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --contour --compare --contour-values 0.5,1.0,2.0,4.0

  Compare three slice files using fixed contour values:
    python tools/replot_slice_data.py data/run1/SampledData0_slices.h5 data/run2/SampledData0_slices.h5 data/run3/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --contour --compare --contour-values 0.5,1.0,2.0,4.0

  Compare normalized vorticity using fixed normalized contour values:
    python tools/replot_slice_data.py data/run1/SampledData0_slices.h5 data/run2/SampledData0_slices.h5 --field vorticity_magnitude --slice xy_center --normalize vorticity --contour --compare --contour-values 0.5,1.0,2.0
"""

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

COMPARE_COLORS = ("k", "red", "blue")


def _yt_font_style():
    return {
        "family": "serif",
        "size": 18,
    }


def _prepare_plot_values(values):
    """Clean saved slice values before plotting."""
    values = np.asarray(values, dtype=np.float64).copy()
    values[np.abs(values) < 1.0e-12] = 0.0
    return np.round(values, decimals=10)


def _star_plot_label(plot_label):
    """Append a superscript star to a plot label."""
    plot_label = str(plot_label)
    if plot_label.startswith("$") and plot_label.endswith("$"):
        core = plot_label[1:-1]
        if "^*" in core:
            return plot_label
        if core.startswith("|") and core.endswith("|"):
            return f"${core[:-1]}^*|$"
        return f"${core}^*$"
    if plot_label.endswith("^*"):
        return plot_label
    return f"{plot_label}^*"


def _apply_normalization(saved, normalize, print_stats=False):
    """Return a copy of saved slice data with optional normalization applied."""
    values_raw = _prepare_plot_values(saved["values"])
    attrs = dict(saved["attrs"])
    values = values_raw.copy()

    if print_stats:
        print(f"Raw data min: {float(np.min(values_raw)):.6g}")
        print(f"Raw data max: {float(np.max(values_raw)):.6g}")

    if normalize == "none":
        normalized = dict(saved)
        normalized["values"] = values
        normalized["attrs"] = attrs
        return normalized

    if normalize == "vorticity":
        field_family = str(attrs.get("field_family", "")).strip()
        if field_family != "vorticity":
            if print_stats:
                print(
                    f"Normalization preset '{normalize}' requested, but field '{attrs.get('field_name', '')}' "
                    f"is not in the vorticity family. Leaving values unchanged."
                )
            normalized = dict(saved)
            normalized["values"] = values
            normalized["attrs"] = attrs
            return normalized

        U0 = 1.0
        L = 1.0 / (2.0 * np.pi)
        reference_scale = U0 / L
        normalization_factor = 1.0 / reference_scale
        values = np.round(values_raw * normalization_factor, decimals=10)
        attrs["plot_label"] = _star_plot_label(attrs.get("plot_label", attrs.get("field_name", "")))
        attrs["normalization"] = normalize
        attrs["normalization_factor"] = normalization_factor
        attrs["normalization_reference_scale"] = reference_scale

        if print_stats:
            print("Applying normalization preset: vorticity")
            print(f"  U0 = {U0:.6g}")
            print(f"  L = {L:.6g}")
            print(f"  Reference scale U0/L = {reference_scale:.6g}")
            print("  Units check: omega has units 1/T and U0/L has units 1/T.")
            print(f"  Normalization uses omega* = omega / (U0/L) = omega * {normalization_factor:.6g}")
            print(f"Normalized data min: {float(np.min(values)):.6g}")
            print(f"Normalized data max: {float(np.max(values)):.6g}")

        normalized = dict(saved)
        normalized["values"] = values
        normalized["attrs"] = attrs
        return normalized

    raise ValueError(f"Unsupported normalization preset: {normalize}")


def _contour_limits(values, vmin, vmax):
    """Return contour range and validate it is not degenerate."""
    zmin = float(np.min(values) if vmin is None else vmin)
    zmax = float(np.max(values) if vmax is None else vmax)
    if np.isclose(zmin, zmax):
        raise ValueError("Contour plotting requires non-constant slice values.")
    return zmin, zmax


def _parse_contour_values(contour_values):
    """Parse a comma-separated list of contour values."""
    if contour_values is None:
        return None

    pieces = [piece.strip() for piece in str(contour_values).split(",")]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        raise ValueError("--contour-values must contain at least one numeric value.")

    try:
        levels = np.array([float(piece) for piece in pieces], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("--contour-values must be a comma-separated list of numbers.") from exc

    levels = np.unique(levels)
    if levels.size == 0:
        raise ValueError("--contour-values must contain at least one numeric value.")
    return np.sort(levels)


def _resolve_contour_levels(values, vmin, vmax, contour_levels, contour_values):
    """Return explicit contour levels from user input or default spacing."""
    if contour_values is not None:
        return _parse_contour_values(contour_values)

    zmin, zmax = _contour_limits(values, vmin, vmax)
    return np.linspace(zmin, zmax, contour_levels)


def _build_contour_grid(horizontal_coords, vertical_coords, values, target_size=None):
    """Return a regular grid for smoother contouring on yt-backed plots."""
    horizontal_coords = np.asarray(horizontal_coords, dtype=np.float64)
    vertical_coords = np.asarray(vertical_coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if target_size is None:
        target_size = min(max(8 * max(values.shape), 512), 2048)

    target_h = max(len(horizontal_coords), int(target_size))
    target_v = max(len(vertical_coords), int(target_size))

    if target_h == len(horizontal_coords) and target_v == len(vertical_coords):
        X, Y = np.meshgrid(horizontal_coords, vertical_coords)
        return X, Y, values

    dense_h = np.linspace(float(horizontal_coords[0]), float(horizontal_coords[-1]), target_h)
    dense_v = np.linspace(float(vertical_coords[0]), float(vertical_coords[-1]), target_v)

    try:
        from scipy.interpolate import RectBivariateSpline

        ky = min(3, len(vertical_coords) - 1)
        kx = min(3, len(horizontal_coords) - 1)
        if ky >= 1 and kx >= 1:
            spline = RectBivariateSpline(vertical_coords, horizontal_coords, values, ky=ky, kx=kx)
            dense_values = spline(dense_v, dense_h)
        else:
            raise ValueError("Not enough points for spline interpolation.")
    except Exception:
        dense_values = values
        dense_h = horizontal_coords
        dense_v = vertical_coords

    X, Y = np.meshgrid(dense_h, dense_v)
    return X, Y, dense_values


def _comparison_label(slice_file, field_name, saved_attrs):
    """Return a short label for one compared dataset."""
    source = output_source_path(slice_file, field_name, saved_attrs)
    return os.path.splitext(os.path.basename(source))[0]


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


def output_stem(source_path, field_name):
    """Build a clean default filename stem with the plotted field name once in front."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    marker = "_sampled_data"
    if marker in base:
        _, suffix = base.split(marker, 1)
        return f"{field_name}{marker}{suffix}"
    if base == field_name or base.startswith(f"{field_name}_"):
        return base
    if base.endswith(f"_{field_name}"):
        return base[: -(len(field_name) + 1)]
    return f"{field_name}_{base}"


def normalization_suffix(normalize):
    """Return a filename suffix for a normalization preset."""
    if normalize == "none":
        return ""
    if normalize == "vorticity":
        return "_vorticity_star"
    return f"_{normalize}"


def output_source_path(slice_file, field_name, saved_attrs):
    """Return the path whose stem should drive default replot naming for one field."""
    source_h5 = str(saved_attrs.get("source_h5", "")).strip()
    source_file = str(saved_attrs.get("source_file", "")).strip()
    field_family = str(saved_attrs.get("field_family", "")).strip()
    source_dataset = str(saved_attrs.get("source_dataset", field_name)).strip()

    if field_family == "scalar" and source_h5 and os.path.exists(source_h5):
        with h5py.File(source_h5, "r") as hf:
            if "fields" in hf and source_dataset in hf["fields"]:
                dataset = hf["fields"][source_dataset]
                for attr_name in ("source_path", "source_h5", "source_txt"):
                    candidate = dataset.attrs.get(attr_name)
                    if candidate:
                        return str(candidate)

    for candidate in (source_file, source_h5):
        if candidate:
            return candidate
    return slice_file


def default_output_directory(slice_file):
    """Return the directory where replots should be written."""
    slice_path = Path(slice_file).resolve()
    if slice_path.parent.name == "slice_data":
        output_dir = slice_path.parent.parent / "slice_plots" / "slice_replots"
    else:
        output_dir = slice_path.parent / "slice_replots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def default_output_name(slice_file, field_name, slice_tag, output_format, saved_attrs, normalize="none"):
    """Return the default output path for one replotted saved slice."""
    output_dir = default_output_directory(slice_file)
    base = output_stem(output_source_path(slice_file, field_name, saved_attrs), field_name) + normalization_suffix(normalize)
    return os.path.join(output_dir, f"{base}_{slice_tag}.{output_format}")


def default_comparison_output_name(slice_file, field_name, slice_tag, output_format, saved_attrs, normalize="none"):
    """Return the default output path for one copied comparison plot."""
    return default_output_name(
        slice_file,
        field_name,
        f"{slice_tag}_contour_compare",
        output_format,
        saved_attrs,
        normalize=normalize,
    )


def default_comparison_metadata_name(slice_file, field_name, slice_tag, saved_attrs, normalize="none"):
    """Return the default output path for one copied comparison metadata file."""
    output_path = default_comparison_output_name(
        slice_file,
        field_name,
        slice_tag,
        "pdf",
        saved_attrs,
        normalize=normalize,
    )
    stem, _ = os.path.splitext(output_path)
    return f"{stem}_metadata.txt"


def build_comparison_metadata_text(prepared, field_name, slice_tag, normalize, levels):
    """Return metadata text describing one comparison contour plot."""
    lines = [
        f"Field: {field_name}",
        f"Slice: {slice_tag}",
        f"Normalization: {normalize}",
        "Contour colors:",
    ]
    for index, item in enumerate(prepared):
        color = item["compare_color"]
        lines.append(f"  {color}: {os.path.abspath(item['slice_file'])}")
    lines.append("Contour levels:")
    lines.append("  " + ", ".join(f"{float(level):.6g}" for level in np.asarray(levels, dtype=np.float64)))
    return "\n".join(lines) + "\n"


def build_yt_slice_dataset(slice_file, field_name, slice_tag, saved=None):
    """Construct a one-cell-thick yt uniform-grid dataset from one saved slice."""
    import yt

    if saved is None:
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


def render_saved_slice_contour_matplotlib(
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
    contour_levels,
    contour_values,
    contour_filled,
    contour_color,
    normalize,
):
    """Render a separate contour plot for one saved slice using matplotlib.

    This is always a second output file alongside the regular yt colormap plot.
    Uses matplotlib's marching-squares contour algorithm directly on the raw 2D
    data array, producing smooth contour curves independent of buffer resolution.
    """
    import matplotlib.pyplot as plt

    saved = _apply_normalization(load_saved_slice(slice_file, field_name, slice_tag), normalize, print_stats=False)
    attrs = saved["attrs"]
    values = _prepare_plot_values(saved["values"])
    h_coords = np.asarray(saved["coord_horizontal"], dtype=np.float64)
    v_coords = np.asarray(saved["coord_vertical"], dtype=np.float64)

    levels = _resolve_contour_levels(values, vmin, vmax, contour_levels, contour_values)

    X, Y = np.meshgrid(h_coords, v_coords)

    fig, ax = plt.subplots(figsize=(figure_size, figure_size))

    if contour_filled:
        cf = ax.contourf(X, Y, values, levels=levels, cmap=cmap)
        fig.colorbar(cf, ax=ax, label=str(attrs["plot_label"]))

    CS = ax.contour(X, Y, values, levels=levels, colors=contour_color, linewidths=1.5)
    ax.clabel(CS, fontsize=8, inline=True, fmt="%.3g")

    h_axis = str(attrs.get("horizontal_axis", "h"))
    v_axis = str(attrs.get("vertical_axis", "v"))
    ax.set_xlabel(h_axis)
    ax.set_ylabel(v_axis)
    ax.set_title(f"{field_name}  [{slice_tag}]")
    ax.set_aspect("equal")
    if width is not None:
        half_width = 0.5 * float(width)
        h_center = 0.5 * (float(h_coords[0]) + float(h_coords[-1]))
        v_center = 0.5 * (float(v_coords[0]) + float(v_coords[-1]))
        ax.set_xlim(h_center - half_width, h_center + half_width)
        ax.set_ylim(v_center - half_width, v_center + half_width)
    fig.tight_layout()

    output_path = output or default_output_name(
        slice_file,
        field_name,
        f"{slice_tag}_contour",
        output_format,
        attrs,
        normalize=normalize,
    )
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved contour: {output_path}")
    if plot:
        plt.show()
    plt.close(fig)


def render_saved_slice_contour_yt(
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
    contour_levels,
    contour_values,
    contour_filled,
    contour_color,
    normalize,
):
    """Render a separate contour plot for one saved slice using yt."""
    import yt

    saved = _apply_normalization(load_saved_slice(slice_file, field_name, slice_tag), normalize, print_stats=False)
    dataset, yt_field, saved = build_yt_slice_dataset(slice_file, field_name, slice_tag, saved=saved)
    values = _prepare_plot_values(saved["values"])
    attrs = saved["attrs"]
    horizontal_coords = np.asarray(saved["coord_horizontal"], dtype=np.float64)
    vertical_coords = np.asarray(saved["coord_vertical"], dtype=np.float64)
    output_path = output or default_output_name(
        slice_file,
        field_name,
        f"{slice_tag}_contour",
        output_format,
        attrs,
        normalize=normalize,
    )
    zmin, zmax = _contour_limits(values, vmin, vmax)
    levels = _resolve_contour_levels(values, vmin, vmax, contour_levels, contour_values)
    grid_x, grid_y, contour_field = _build_contour_grid(horizontal_coords, vertical_coords, values)

    slice_plot = yt.SlicePlot(dataset, str(attrs["axis"]), yt_field, center="c", origin="native")
    slice_plot.set_log(yt_field, False)
    slice_plot.set_cmap(yt_field, cmap)
    slice_plot.set_axes_unit("unitary")
    slice_plot.set_font(_yt_font_style())
    slice_plot.set_figure_size(float(figure_size))
    data_res = max(values.shape)
    slice_plot.set_buff_size((data_res, data_res))
    slice_plot.set_zlim(yt_field, zmin=zmin, zmax=zmax)
    slice_plot.set_minorticks("all", False)
    slice_plot.set_colorbar_minorticks("all", False)
    if width is not None:
        slice_plot.set_width((float(width), "code_length"))

    if not contour_filled:
        slice_plot.hide_colorbar()

    slice_plot.render()
    window_plot = slice_plot.plots[yt_field]
    image = window_plot.axes.images[0]
    image.set_interpolation("bicubic")
    if hasattr(image, "set_interpolation_stage"):
        image.set_interpolation_stage("rgba")
    axes = window_plot.axes
    image.set_visible(False)

    if contour_filled:
        axes.contourf(grid_x, grid_y, contour_field, levels=levels, cmap=cmap)
        window_plot.cb.set_label(str(attrs["plot_label"]))

    contour_set = axes.contour(grid_x, grid_y, contour_field, levels=levels, colors=contour_color, linewidths=1.5)
    axes.clabel(contour_set, fontsize=8, inline=True, fmt="%.3g")
    saved_path = window_plot.save(output_path, mpl_kwargs={"dpi": dpi})
    print(f"Saved contour: {saved_path}")
    if plot:
        slice_plot.show()


def render_saved_slice_contour(
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
    contour_levels,
    contour_values,
    contour_filled,
    contour_color,
    contour_backend,
    normalize,
):
    """Render a separate contour plot using yt or matplotlib."""
    if contour_backend == "matplotlib":
        render_saved_slice_contour_matplotlib(
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
            contour_levels,
            contour_values,
            contour_filled,
            contour_color,
            normalize,
        )
        return

    try:
        render_saved_slice_contour_yt(
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
            contour_levels,
            contour_values,
            contour_filled,
            contour_color,
            normalize,
        )
    except Exception as exc:
        if contour_backend == "yt":
            raise
        print(f"yt contour rendering failed ({exc}); falling back to matplotlib.")
        render_saved_slice_contour_matplotlib(
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
            contour_levels,
            contour_values,
            contour_filled,
            contour_color,
            normalize,
        )


def render_compared_contours(
    slice_files,
    field_name,
    slice_tag,
    cmap,
    width,
    vmin,
    vmax,
    output_format,
    plot,
    dpi,
    figure_size,
    contour_levels,
    contour_values,
    contour_color,
    normalize,
    contour_filled=False,
    output_paths=None,
    metadata_paths=None,
):
    """Render one overlaid comparison contour plot for up to three slice files."""
    import yt

    if len(slice_files) < 2:
        raise ValueError("--compare requires at least two slice files.")
    if len(slice_files) > len(COMPARE_COLORS):
        raise ValueError("--compare supports at most three slice files.")

    if contour_filled:
        print("Compare mode ignores --contour-filled and renders line contours only.")

    prepared = []
    reference_horizontal = None
    reference_vertical = None
    reference_axes = None

    for slice_file in slice_files:
        saved = _apply_normalization(load_saved_slice(slice_file, field_name, slice_tag), normalize, print_stats=False)
        attrs = saved["attrs"]
        values = _prepare_plot_values(saved["values"])
        horizontal_coords = np.asarray(saved["coord_horizontal"], dtype=np.float64)
        vertical_coords = np.asarray(saved["coord_vertical"], dtype=np.float64)

        if reference_horizontal is None:
            reference_horizontal = horizontal_coords
            reference_vertical = vertical_coords
            reference_axes = (
                str(attrs.get("horizontal_axis", "h")),
                str(attrs.get("vertical_axis", "v")),
            )
        else:
            if (
                horizontal_coords.shape != reference_horizontal.shape
                or vertical_coords.shape != reference_vertical.shape
                or not np.allclose(horizontal_coords, reference_horizontal)
                or not np.allclose(vertical_coords, reference_vertical)
            ):
                raise ValueError("All compared slice files must share the same plotting coordinates.")
            current_axes = (
                str(attrs.get("horizontal_axis", "h")),
                str(attrs.get("vertical_axis", "v")),
            )
            if current_axes != reference_axes:
                raise ValueError("All compared slice files must share the same displayed axes.")

        grid_x, grid_y, contour_field = _build_contour_grid(horizontal_coords, vertical_coords, values)
        prepared.append(
            {
                "slice_file": slice_file,
                "attrs": attrs,
                "saved": saved,
                "values": values,
                "horizontal_coords": horizontal_coords,
                "vertical_coords": vertical_coords,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "contour_field": contour_field,
                "label": _comparison_label(slice_file, field_name, attrs),
            }
        )

    if contour_values is not None:
        levels = _parse_contour_values(contour_values)
    else:
        global_min = min(float(np.min(item["values"])) for item in prepared)
        global_max = max(float(np.max(item["values"])) for item in prepared)
        levels = _resolve_contour_levels(
            np.array([global_min, global_max], dtype=np.float64),
            vmin,
            vmax,
            contour_levels,
            None,
        )

    base_item = prepared[0]
    base_dataset, base_yt_field, _ = build_yt_slice_dataset(
        base_item["slice_file"],
        field_name,
        slice_tag,
        saved=base_item["saved"],
    )
    base_min = float(np.min(base_item["values"]))
    base_max = float(np.max(base_item["values"]))
    zmin = float(base_min if vmin is None else vmin)
    zmax = float(base_max if vmax is None else vmax)

    slice_plot = yt.SlicePlot(
        base_dataset,
        str(base_item["attrs"]["axis"]),
        base_yt_field,
        center="c",
        origin="native",
    )
    slice_plot.set_log(base_yt_field, False)
    slice_plot.set_cmap(base_yt_field, cmap)
    slice_plot.set_axes_unit("unitary")
    slice_plot.set_font(_yt_font_style())
    slice_plot.set_figure_size(float(figure_size))
    data_res = max(base_item["values"].shape)
    slice_plot.set_buff_size((data_res, data_res))
    if not np.isclose(zmin, zmax):
        slice_plot.set_zlim(base_yt_field, zmin=zmin, zmax=zmax)
    slice_plot.set_minorticks("all", False)
    slice_plot.set_colorbar_minorticks("all", False)
    if width is not None:
        slice_plot.set_width((float(width), "code_length"))
    slice_plot.hide_colorbar()

    slice_plot.render()
    window_plot = slice_plot.plots[base_yt_field]
    image = window_plot.axes.images[0]
    image.set_interpolation("bicubic")
    if hasattr(image, "set_interpolation_stage"):
        image.set_interpolation_stage("rgba")
    image.set_visible(False)
    ax = window_plot.axes

    for index, item in enumerate(prepared):
        color = contour_color if index == 0 else COMPARE_COLORS[index]
        item["compare_color"] = color
        ax.contour(
            item["grid_x"],
            item["grid_y"],
            item["contour_field"],
            levels=levels,
            colors=color,
            linewidths=1.5,
        )
        print(f"Compare color: {item['label']} -> {color}")

    h_axis, v_axis = reference_axes
    ax.set_xlabel(h_axis)
    ax.set_ylabel(v_axis)
    ax.set_aspect("equal")
    if width is not None:
        half_width = 0.5 * float(width)
        h_center = 0.5 * (float(reference_horizontal[0]) + float(reference_horizontal[-1]))
        v_center = 0.5 * (float(reference_vertical[0]) + float(reference_vertical[-1]))
        ax.set_xlim(h_center - half_width, h_center + half_width)
        ax.set_ylim(v_center - half_width, v_center + half_width)

    if output_paths is None:
        output_paths = [
            default_comparison_output_name(
                item["slice_file"],
                field_name,
                slice_tag,
                output_format,
                item["attrs"],
                normalize=normalize,
            )
            for item in prepared
        ]
    if metadata_paths is None:
        metadata_paths = [
            default_comparison_metadata_name(
                item["slice_file"],
                field_name,
                slice_tag,
                item["attrs"],
                normalize=normalize,
            )
            for item in prepared
        ]

    for output_path in dict.fromkeys(output_paths):
        saved_path = window_plot.save(output_path, mpl_kwargs={"dpi": dpi})
        print(f"Saved comparison contour: {saved_path}")
    metadata_text = build_comparison_metadata_text(prepared, field_name, slice_tag, normalize, levels)
    for metadata_path in dict.fromkeys(metadata_paths):
        with open(metadata_path, "w", encoding="utf-8") as handle:
            handle.write(metadata_text)
        print(f"Saved comparison metadata: {metadata_path}")

    if plot:
        slice_plot.show()


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
    normalize,
):
    """Render one saved slice plane to disk and/or screen with yt bicubic output."""
    import yt

    saved = _apply_normalization(load_saved_slice(slice_file, field_name, slice_tag), normalize, print_stats=True)
    dataset, yt_field, saved = build_yt_slice_dataset(slice_file, field_name, slice_tag, saved=saved)
    values = _prepare_plot_values(saved["values"])

    attrs = saved["attrs"]
    output_path = output or default_output_name(
        slice_file,
        field_name,
        slice_tag,
        output_format,
        attrs,
        normalize=normalize,
    )
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    print(f"Plot data min: {data_min:.6g}")
    print(f"Plot data max: {data_max:.6g}")
    zmin = float(data_min if vmin is None else vmin)
    zmax = float(data_max if vmax is None else vmax)

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
#     output_path = output or default_output_name(slice_file, field_name, slice_tag, output_format, saved["attrs"])
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


def process_slice_file(slice_file, args):
    """Process one slice file using parsed CLI arguments."""
    print(f"Processing slice file: {os.path.abspath(slice_file)}")

    if args.list:
        print_summary(slice_file)
        return

    if args.field is None and args.slice_tag is None:
        print_summary(slice_file)
        return

    print_saved_slice_metadata(slice_file, args.field, args.slice_tag)

    if args.yt_info:
        print_yt_summary(slice_file, args.field, args.slice_tag)

    render_saved_slice(
        slice_file,
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
        args.normalize,
    )

    if args.contour:
        render_saved_slice_contour(
            slice_file,
            args.field,
            args.slice_tag,
            args.cmap,
            args.width,
            args.vmin,
            args.vmax,
            None,
            args.format,
            args.plot,
            args.dpi,
            args.figsize,
            args.contour_levels,
            args.contour_values,
            args.contour_filled,
            args.contour_color,
            args.contour_backend,
            args.normalize,
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect and replot saved 2D slice data from *_slices.h5 files.")
    parser.add_argument("slice_file", nargs="+", help="One or more paths to combined *_slices.h5 files")
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
        "--normalize",
        default="none",
        choices=["none", "vorticity"],
        help="Optional normalization preset. 'vorticity' uses U0=1 and L=1/(2*pi), and interprets contour values in normalized units.",
    )
    parser.add_argument(
        "--yt-info",
        action="store_true",
        help="Construct a one-cell-thick yt uniform-grid dataset for the selected slice and print its summary.",
    )
    parser.add_argument("--contour", action="store_true", help="Also render a separate contour plot alongside the default yt colormap plot.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Overlay contour lines from two or three slice files and copy the comparison plot into each slice_replots directory.",
    )
    parser.add_argument("--contour-levels", type=int, default=12, help="Number of contour levels. Default is 12.")
    parser.add_argument(
        "--contour-values",
        default=None,
        help="Optional comma-separated contour values, for example '0.5,1.0,2.0,4.0,8.0'. Overrides --contour-levels.",
    )
    parser.add_argument("--contour-filled", action="store_true", help="Use filled contours (contourf) as background behind the contour lines.")
    parser.add_argument("--contour-color", default="k", help="Single color for contour lines, e.g. 'k', 'white', '#ff0000'. Default is 'k' (black).")
    parser.add_argument(
        "--contour-backend",
        default="yt",
        choices=["auto", "yt", "matplotlib"],
        help="Contour renderer to use. Default is 'yt'. 'auto' tries yt first and falls back to matplotlib.",
    )
    args = parser.parse_args()

    if args.field is None or args.slice_tag is None:
        if not (args.field is None and args.slice_tag is None):
            raise SystemExit("--field and --slice must be provided together.")
    if args.output is not None and len(args.slice_file) > 1:
        raise SystemExit("--output can only be used when processing a single slice file.")
    if args.compare and not args.list and not (args.field is None and args.slice_tag is None):
        if not args.contour:
            raise SystemExit("--compare requires --contour.")
        if len(args.slice_file) < 2:
            raise SystemExit("--compare requires at least two slice files.")
        if len(args.slice_file) > len(COMPARE_COLORS):
            raise SystemExit("--compare supports at most three slice files.")

    for index, slice_file in enumerate(args.slice_file):
        if index > 0:
            print()
            print("=" * 80)
            print()
        process_slice_file(slice_file, args)

    if args.compare and not args.list and args.field is not None and args.slice_tag is not None:
        print()
        print("=" * 80)
        print()
        print("Rendering comparison contour overlay...")
        render_compared_contours(
            args.slice_file,
            args.field,
            args.slice_tag,
            args.cmap,
            args.width,
            args.vmin,
            args.vmax,
            args.format,
            args.plot,
            args.dpi,
            args.figsize,
            args.contour_levels,
            args.contour_values,
            args.contour_color,
            args.normalize,
            contour_filled=args.contour_filled,
        )


if __name__ == "__main__":
    main()
