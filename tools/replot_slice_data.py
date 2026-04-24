#!/usr/bin/env python3
"""Inspect and replot saved 2D slice data from a combined *_slices.h5 file.

Example commands:
  Notes:
    --field selects the saved variable to replot using:
      1 -> vorticity_magnitude
      2 -> velocity_magnitude
      3 -> the saved scalar field when there is exactly one scalar field
      q_criterion, r_criterion, and density_gradient_magnitude are available by explicit field name when present
    --slice selects which saved plane to render using:
      1 -> xy_center
      2 -> xy_face
      3 -> yz_face
      4 -> zx_face
    If --slice is omitted, all saved slices for the selected field are rendered.
    If more than one slice file is provided, comparison contour overlays are rendered automatically.
    When contour plots are rendered, matching zoomed contour plots are also written using --zoom-window.

  List available fields and slices:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --list

  Render one yt slice image:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field 1 --slice 1

  Render one slice image plus contour and zoomed contour plots:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field 1 --slice 1 --contour

  Render contours with a fixed number of levels:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field 1 --slice 1 --contour --contour-levels 20

  Render contours at explicit values:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field 1 --slice 1 --contour --contour-values 0.5,1.0,2.0,4.0,8.0

  Render normalized vorticity with interpolation disabled:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field 1 --slice 1 --normalize --contour --contour-values 0.5,1.0,2.0 --interpolate 0

  Render every saved slice for one field into per-slice subdirectories:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field 1

  Render every saved slice with contours:
    python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --field 1 --contour

  Compare two or three slice files automatically:
    python tools/replot_slice_data.py data/run1/SampledData0_slices.h5 data/run2/SampledData0_slices.h5 data/run3/SampledData0_slices.h5 --field 1 --slice 1

  Compare two slice files using fixed contour values:
    python tools/replot_slice_data.py data/run1/SampledData0_slices.h5 data/run2/SampledData0_slices.h5 --field 1 --slice 1 --contour-values 0.5,1.0,2.0,4.0

  Compare all saved slices with fixed contour values and a custom zoom window:
    python tools/replot_slice_data.py data/run1/SampledData0_slices.h5 data/run2/SampledData0_slices.h5 --field 1 --contour-values 0.5,1.0,2.0,4.0 --zoom-window 0.0,0.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from postprocess_vis.normalization_labels import format_plot_label

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPARE_COLORS = ("k","red","green")
CONTOUR_LABEL_COLOR = "black"
CONTOUR_LABEL_FONTSIZE = 8
FIELD_SHORTCUTS = {
    "1": "vorticity_magnitude",
    "2": "velocity_magnitude",
}
SLICE_SHORTCUTS = {
    "1": "xy_center",
    "2": "xy_face",
    "3": "yz_face",
    "4": "zx_face",
}
# Saved slice selectors:
#   1 -> xy_center
#   2 -> xy_face
#   3 -> yz_face
#   4 -> zx_face

DEFAULT_ZOOM_WINDOW = (0.0, 0.5)


def _yt_font_style():
    return {
        "family": "serif",
        "size": 18,
    }


def _ensure_yt_imported():
    """Import yt early so cluster library resolution is stable before h5py helpers."""
    import yt  # noqa: F401


def _list_available_slices(filepath):
    from postprocess_vis.slice_data import list_available_slices

    return list_available_slices(filepath)


def _field_family_map(filepath):
    import h5py

    with h5py.File(filepath, "r") as hf:
        slices_group = hf["slices"]
        return {
            field_name: str(slices_group[field_name].attrs.get("field_family", "")).strip()
            for field_name in slices_group.keys()
        }


def _resolve_field_name(filepath, field_name):
    """Resolve numeric field shortcuts to stored slice field names."""
    if field_name is None:
        return None

    field_name = str(field_name).strip()
    if field_name in FIELD_SHORTCUTS:
        return FIELD_SHORTCUTS[field_name]

    if field_name == "3":
        family_map = _field_family_map(filepath)
        scalar_fields = sorted(name for name, family in family_map.items() if family == "scalar")
        if len(scalar_fields) == 1:
            return scalar_fields[0]
        if not scalar_fields:
            raise SystemExit("Field selector '3' requested a scalar field, but no scalar field was saved in this slice file.")
        raise SystemExit(
            "Field selector '3' is ambiguous because multiple scalar fields were saved: "
            + ", ".join(scalar_fields)
            + ". Use the explicit field name instead."
        )

    return field_name


def _resolve_slice_name(slice_tag):
    """Resolve numeric slice shortcuts to stored slice tags."""
    if slice_tag is None:
        return None

    slice_tag = str(slice_tag).strip()
    if slice_tag in SLICE_SHORTCUTS:
        return SLICE_SHORTCUTS[slice_tag]
    return slice_tag


def _load_saved_slice(filepath, field_name, slice_tag):
    from postprocess_vis.slice_data import load_saved_slice

    return load_saved_slice(filepath, field_name, slice_tag)


def _available_slice_tags(filepath, field_name):
    summary = _list_available_slices(filepath)
    fields = summary["fields"]
    if field_name not in fields:
        raise SystemExit(
            f"Field '{field_name}' is not available in {os.path.abspath(filepath)}. "
            f"Available fields: {', '.join(sorted(fields))}"
        )
    return list(fields[field_name])


def _resolve_slice_tags(filepath, field_name, slice_tag):
    """Resolve one requested slice tag or all saved tags for a field."""
    available_tags = _available_slice_tags(filepath, field_name)
    if slice_tag is None:
        return available_tags
    if slice_tag not in available_tags:
        raise SystemExit(
            f"Slice '{slice_tag}' is not available for field '{field_name}' in {os.path.abspath(filepath)}. "
            f"Available slices: {', '.join(available_tags)}"
        )
    return [slice_tag]


def _plane_extent_from_arrays(horizontal_coords, vertical_coords):
    from postprocess_vis.slice_data import plane_extent_from_arrays

    return plane_extent_from_arrays(horizontal_coords, vertical_coords)


def _prepare_plot_values(values):
    """Clean saved slice values before plotting."""
    values = np.asarray(values, dtype=np.float64).copy()
    values[np.abs(values) < 1.0e-12] = 0.0
    return np.round(values, decimals=10)


def _stored_global_limits(attrs):
    """Return stored full-volume colorbar limits when present."""
    if "global_min" not in attrs or "global_max" not in attrs:
        return None
    return float(attrs["global_min"]), float(attrs["global_max"])


def _resolve_saved_color_limits(saved, vmin=None, vmax=None):
    """Resolve the colorbar limits used for one saved slice plot."""
    values = _prepare_plot_values(saved["values"])
    attrs = saved["attrs"]
    stored_limits = _stored_global_limits(attrs)
    if stored_limits is None:
        data_min = float(np.min(values))
        data_max = float(np.max(values))
        source = "gathered 2D slice fallback"
    else:
        data_min, data_max = stored_limits
        source = "stored global 3D limits"

    zmin = float(data_min if vmin is None else vmin)
    zmax = float(data_max if vmax is None else vmax)
    return data_min, data_max, zmin, zmax, source


def _parse_zoom_window(zoom_window):
    """Parse a zoom window as xmin,xmax or xmin,xmax,ymin,ymax."""
    if zoom_window is None:
        return None

    if isinstance(zoom_window, tuple) and len(zoom_window) in {2, 4}:
        return tuple(float(value) for value in zoom_window)

    pieces = [piece.strip() for piece in str(zoom_window).split(",") if piece.strip()]
    try:
        values = tuple(float(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError("--zoom-window must be a comma-separated list of numbers.") from exc

    if len(values) not in {2, 4}:
        raise ValueError("--zoom-window must contain either 2 values (xmin,xmax) or 4 values (xmin,xmax,ymin,ymax).")
    return values


def _apply_zoom_limits(ax, zoom_window):
    """Apply explicit axis limits for a zoomed contour view."""
    if zoom_window is None:
        return

    if len(zoom_window) == 2:
        xmin, xmax = zoom_window
        ymin, ymax = zoom_window
    else:
        xmin, xmax, ymin, ymax = zoom_window

    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))


def _apply_image_interpolation(image, interpolate):
    """Apply visual interpolation to a yt-backed image plot."""
    if interpolate:
        image.set_interpolation("bicubic")
        if hasattr(image, "set_interpolation_stage"):
            image.set_interpolation_stage("rgba")
        return

    image.set_interpolation("nearest")
    if hasattr(image, "set_interpolation_stage"):
        image.set_interpolation_stage("data")


def _comparison_contour_label_color(num_datasets):
    """Return the contour-label color for compare mode."""
    if int(num_datasets) >= 3:
        return "green"
    return CONTOUR_LABEL_COLOR


def _style_contour_labels(labels):
    """Apply shared styling to contour labels."""
    for label in labels:
        label.set_fontweight("bold")
        label.set_bbox(
            {
                "facecolor": "white",
                "edgecolor": "white",
                "boxstyle": "square,pad=0.1",
            }
        )


def _normalization_mode(normalize_requested, attrs):
    """Infer the normalization mode from the field family."""
    if not normalize_requested:
        return "none"

    field_family = str(attrs.get("field_family", "")).strip()
    if field_family == "vorticity":
        return "vorticity"
    if field_family == "velocity":
        return "velocity"
    return "none"


def _value_normalization_scale(mode, attrs):
    """Return the reference scale for one saved-value normalization mode."""
    resolved = str(mode or "none").strip().lower()
    if resolved == "none":
        return 1.0
    if resolved == "global_rms":
        rms = float(attrs.get("global_rms", 0.0))
        return rms if abs(rms) > 1.0e-30 else None
    return None


def _resolve_value_normalization_mode(requested_mode, attrs):
    """Resolve a requested saved-value normalization against the file metadata."""
    saved_mode = str(attrs.get("value_normalization", "none")).strip().lower()
    requested = str(requested_mode or "saved").strip().lower()
    if requested == "saved":
        return saved_mode
    return requested


def _display_normalization_label(value_normalization, extra_normalization):
    """Return one combined normalization description string."""
    modes = []
    value_mode = str(value_normalization or "none").strip()
    extra_mode = str(extra_normalization or "none").strip()
    if value_mode and value_mode != "none":
        modes.append(value_mode)
    if extra_mode and extra_mode != "none":
        modes.append(extra_mode)
    if not modes:
        return "none"
    return "+".join(modes)


def _apply_normalization(saved, value_normalization_mode, normalize, print_stats=False):
    """Return a copy of saved slice data with optional normalization applied."""
    values_stored = _prepare_plot_values(saved["values"])
    attrs = dict(saved["attrs"])
    values = values_stored.copy()
    mode = _normalization_mode(normalize, attrs)
    saved_value_normalization = str(attrs.get("value_normalization", "none")).strip().lower()
    value_normalization = _resolve_value_normalization_mode(value_normalization_mode, attrs)
    base_plot_label = attrs.get("base_plot_label", attrs.get("plot_label", attrs.get("field_name", "")))
    stored_limits_saved = _stored_global_limits(attrs)
    saved_scale = _value_normalization_scale(saved_value_normalization, attrs)
    target_scale = _value_normalization_scale(value_normalization, attrs)

    if saved_scale is None:
        saved_scale = 1.0
        saved_value_normalization = "none"
        if print_stats:
            print("Saved slice metadata requested RMS normalization, but no usable global RMS was stored. Treating values as unnormalized.")

    if target_scale is None:
        if print_stats:
            print("Requested RMS normalization, but no usable global RMS was stored. Leaving values unchanged.")
        value_normalization = saved_value_normalization
        target_scale = saved_scale

    conversion_factor = float(saved_scale) / float(target_scale)
    values = np.round(values_stored * conversion_factor, decimals=10)
    stored_limits_raw = None
    if stored_limits_saved is not None:
        stored_limits_raw = (
            float(stored_limits_saved[0] * conversion_factor),
            float(stored_limits_saved[1] * conversion_factor),
        )

    if print_stats:
        if stored_limits_saved is None:
            print("Stored global 3D colorbar limits: unavailable; falling back to slice-local values.")
        else:
            print(f"Stored global 3D colorbar min: {stored_limits_saved[0]:.6g}")
            print(f"Stored global 3D colorbar max: {stored_limits_saved[1]:.6g}")
        if "global_rms" in attrs:
            print(f"Stored global 3D RMS normalization: {float(attrs['global_rms']):.6g}")
        print(f"Saved value normalization: {saved_value_normalization}")
        print(f"Display value normalization: {value_normalization}")

    if mode == "none":
        attrs["plot_label"] = format_plot_label(
            base_plot_label,
            value_normalization=value_normalization,
            extra_normalization="none",
        )
        attrs["saved_value_normalization"] = saved_value_normalization
        attrs["value_normalization"] = value_normalization
        attrs["normalization"] = "none"
        attrs["normalization_factor"] = 1.0
        attrs["display_normalization"] = _display_normalization_label(value_normalization, "none")
        if stored_limits_raw is not None:
            attrs["global_min"] = float(stored_limits_raw[0])
            attrs["global_max"] = float(stored_limits_raw[1])
        if normalize and print_stats:
            print(
                f"Normalization requested, but field '{attrs.get('field_name', '')}' "
                "is neither velocity nor vorticity. Leaving values unchanged."
            )
        if print_stats:
            if stored_limits_raw is None:
                print(f"Using 2D slice fallback min: {float(np.min(values)):.6g}")
                print(f"Using 2D slice fallback max: {float(np.max(values)):.6g}")
            else:
                print(f"Using stored global 3D colorbar min: {stored_limits_raw[0]:.6g}")
                print(f"Using stored global 3D colorbar max: {stored_limits_raw[1]:.6g}")
        normalized = dict(saved)
        normalized["values"] = values
        normalized["attrs"] = attrs
        return normalized

    if mode == "vorticity":
        U0 = 1.0
        L = 1.0 / (2.0 * np.pi)
        reference_scale = U0 / L
        normalization_factor = 1.0 / reference_scale
        values = np.round(values * normalization_factor, decimals=10)
        attrs["plot_label"] = format_plot_label(
            base_plot_label,
            value_normalization=value_normalization,
            extra_normalization=mode,
        )
        attrs["saved_value_normalization"] = saved_value_normalization
        attrs["value_normalization"] = value_normalization
        attrs["normalization"] = mode
        attrs["normalization_factor"] = normalization_factor
        attrs["normalization_reference_scale"] = reference_scale
        attrs["display_normalization"] = _display_normalization_label(value_normalization, mode)
        if stored_limits_raw is not None:
            attrs["global_min"] = float(stored_limits_raw[0] * normalization_factor)
            attrs["global_max"] = float(stored_limits_raw[1] * normalization_factor)

        if print_stats:
            print("Applying normalization: vorticity")
            print(f"  U0 = {U0:.6g}")
            print(f"  L = {L:.6g}")
            print(f"  Reference scale U0/L = {reference_scale:.6g}")
            print("  Units check: omega has units 1/T and U0/L has units 1/T.")
            print(f"  Normalization uses omega* = omega / (U0/L) = omega * {normalization_factor:.6g}")
            if stored_limits_raw is None:
                print(f"Normalized 2D slice fallback min: {float(np.min(values)):.6g}")
                print(f"Normalized 2D slice fallback max: {float(np.max(values)):.6g}")
            else:
                print(f"Normalized global 3D colorbar min: {float(attrs['global_min']):.6g}")
                print(f"Normalized global 3D colorbar max: {float(attrs['global_max']):.6g}")

        normalized = dict(saved)
        normalized["values"] = values
        normalized["attrs"] = attrs
        return normalized

    if mode == "velocity":
        U0 = 1.0
        reference_scale = U0
        normalization_factor = 1.0 / reference_scale
        values = np.round(values * normalization_factor, decimals=10)
        attrs["plot_label"] = format_plot_label(
            base_plot_label,
            value_normalization=value_normalization,
            extra_normalization=mode,
        )
        attrs["saved_value_normalization"] = saved_value_normalization
        attrs["value_normalization"] = value_normalization
        attrs["normalization"] = mode
        attrs["normalization_factor"] = normalization_factor
        attrs["normalization_reference_scale"] = reference_scale
        attrs["display_normalization"] = _display_normalization_label(value_normalization, mode)
        if stored_limits_raw is not None:
            attrs["global_min"] = float(stored_limits_raw[0] * normalization_factor)
            attrs["global_max"] = float(stored_limits_raw[1] * normalization_factor)

        if print_stats:
            print("Applying normalization: velocity")
            print(f"  U0 = {U0:.6g}")
            print("  Units check: |u| has units L/T and U0 has units L/T.")
            print(f"  Normalization uses u* = u / U0 = u * {normalization_factor:.6g}")
            if stored_limits_raw is None:
                print(f"Normalized 2D slice fallback min: {float(np.min(values)):.6g}")
                print(f"Normalized 2D slice fallback max: {float(np.max(values)):.6g}")
            else:
                print(f"Normalized global 3D colorbar min: {float(attrs['global_min']):.6g}")
                print(f"Normalized global 3D colorbar max: {float(attrs['global_max']):.6g}")

        normalized = dict(saved)
        normalized["values"] = values
        normalized["attrs"] = attrs
        return normalized

    raise ValueError(f"Unsupported normalization mode: {mode}")


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


def _build_contour_grid(horizontal_coords, vertical_coords, values, target_size=None, interpolate=True):
    """Return a contour grid, optionally interpolated onto a denser mesh."""
    horizontal_coords = np.asarray(horizontal_coords, dtype=np.float64)
    vertical_coords = np.asarray(vertical_coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if not interpolate:
        X, Y = np.meshgrid(horizontal_coords, vertical_coords)
        return X, Y, values

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
    summary = _list_available_slices(filepath)
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


def normalization_suffix(value_normalization="saved", normalize="none", saved_attrs=None):
    """Return a filename suffix for one combined normalization preset."""
    attrs = saved_attrs or {}
    parts = []

    value_mode = _resolve_value_normalization_mode(value_normalization, attrs)
    if value_mode != "none":
        parts.append(value_mode)

    mode = _normalization_mode(normalize, attrs) if isinstance(normalize, bool) else normalize
    if mode == "vorticity":
        parts.append("vorticity_star")
    elif mode == "velocity":
        parts.append("velocity_star")
    elif mode not in {"", "none", None}:
        parts.append(str(mode))

    if not parts:
        return ""
    return "_" + "_".join(parts)


def output_source_path(slice_file, field_name, saved_attrs):
    """Return the path whose stem should drive default replot naming for one field."""
    import h5py

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


def default_output_directory(slice_file, slice_tag=None):
    """Return the directory where replots should be written."""
    slice_path = Path(slice_file).resolve()
    if slice_path.parent.name == "slice_data":
        output_dir = slice_path.parent.parent / "slice_plots" / "slice_replots"
    else:
        output_dir = slice_path.parent / "slice_replots"
    if slice_tag is not None:
        output_dir = output_dir / str(slice_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def default_output_name(
    slice_file,
    field_name,
    slice_tag,
    output_format,
    saved_attrs,
    value_normalization="saved",
    normalize="none",
    output_tag=None,
):
    """Return the default output path for one replotted saved slice."""
    output_dir = default_output_directory(slice_file, slice_tag=slice_tag)
    base = (
        output_stem(output_source_path(slice_file, field_name, saved_attrs), field_name)
        + normalization_suffix(value_normalization=value_normalization, normalize=normalize, saved_attrs=saved_attrs)
    )
    tag = slice_tag if output_tag is None else output_tag
    return os.path.join(output_dir, f"{base}_{tag}.{output_format}")


def default_comparison_output_name(
    slice_file,
    field_name,
    slice_tag,
    output_format,
    saved_attrs,
    value_normalization="saved",
    normalize="none",
):
    """Return the default output path for one copied comparison plot."""
    return default_output_name(
        slice_file,
        field_name,
        slice_tag,
        output_format,
        saved_attrs,
        value_normalization=value_normalization,
        normalize=normalize,
        output_tag=f"{slice_tag}_contour_compare",
    )


def default_comparison_metadata_name(
    slice_file,
    field_name,
    slice_tag,
    saved_attrs,
    value_normalization="saved",
    normalize="none",
):
    """Return the default output path for one copied comparison metadata file."""
    output_path = default_comparison_output_name(
        slice_file,
        field_name,
        slice_tag,
        "pdf",
        saved_attrs,
        value_normalization=value_normalization,
        normalize=normalize,
    )
    stem, _ = os.path.splitext(output_path)
    return f"{stem}_metadata.txt"


def default_zoom_contour_output_name(
    slice_file,
    field_name,
    slice_tag,
    output_format,
    saved_attrs,
    value_normalization="saved",
    normalize="none",
):
    """Return the default output path for one zoomed contour plot."""
    return default_output_name(
        slice_file,
        field_name,
        slice_tag,
        output_format,
        saved_attrs,
        value_normalization=value_normalization,
        normalize=normalize,
        output_tag=f"{slice_tag}_contour_zoom",
    )


def default_zoom_comparison_output_name(
    slice_file,
    field_name,
    slice_tag,
    output_format,
    saved_attrs,
    value_normalization="saved",
    normalize="none",
):
    """Return the default output path for one zoomed comparison contour plot."""
    return default_output_name(
        slice_file,
        field_name,
        slice_tag,
        output_format,
        saved_attrs,
        value_normalization=value_normalization,
        normalize=normalize,
        output_tag=f"{slice_tag}_contour_compare_zoom",
    )


def default_zoom_comparison_metadata_name(
    slice_file,
    field_name,
    slice_tag,
    saved_attrs,
    value_normalization="saved",
    normalize="none",
):
    """Return the default output path for one zoomed comparison metadata file."""
    output_path = default_zoom_comparison_output_name(
        slice_file,
        field_name,
        slice_tag,
        "pdf",
        saved_attrs,
        value_normalization=value_normalization,
        normalize=normalize,
    )
    stem, _ = os.path.splitext(output_path)
    return f"{stem}_metadata.txt"


def build_comparison_metadata_text(prepared, field_name, slice_tag, normalize, levels):
    """Return metadata text describing one comparison contour plot."""
    normalization_label = "none"
    if prepared:
        normalization_label = str(
            prepared[0]["attrs"].get(
                "display_normalization",
                prepared[0]["attrs"].get("normalization", "none"),
            )
        )
    if prepared:
        contour_min = min(float(item["stored_limits"][0]) for item in prepared)
        contour_max = max(float(item["stored_limits"][1]) for item in prepared)
    else:
        contour_min = contour_max = 0.0
    lines = [
        f"Field: {field_name}",
        f"Slice: {slice_tag}",
        f"Normalization: {normalization_label}",
        f"Stored global contour min: {contour_min:.6g}",
        f"Stored global contour max: {contour_max:.6g}",
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
    import h5py
    import yt

    if saved is None:
        saved = _load_saved_slice(slice_file, field_name, slice_tag)
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
    value_normalization_mode,
    normalize,
    interpolate,
    zoom_window=None,
):
    """Render a separate contour plot for one saved slice using matplotlib.

    This is always a second output file alongside the regular yt colormap plot.
    Uses matplotlib's marching-squares contour algorithm directly on the raw 2D
    data array, producing smooth contour curves independent of buffer resolution.
    """
    import matplotlib.pyplot as plt

    saved = _apply_normalization(
        _load_saved_slice(slice_file, field_name, slice_tag),
        value_normalization_mode,
        normalize,
        print_stats=False,
    )
    attrs = saved["attrs"]
    values = _prepare_plot_values(saved["values"])
    h_coords = np.asarray(saved["coord_horizontal"], dtype=np.float64)
    v_coords = np.asarray(saved["coord_vertical"], dtype=np.float64)
    data_min, data_max, _, _, limits_source = _resolve_saved_color_limits(saved, vmin=vmin, vmax=vmax)
    limits_values = np.array([data_min, data_max], dtype=np.float64)
    levels = _resolve_contour_levels(limits_values, vmin, vmax, contour_levels, contour_values)
    print(f"Contour levels use {limits_source}: min={data_min:.6g}, max={data_max:.6g}")

    X, Y = np.meshgrid(h_coords, v_coords)

    fig, ax = plt.subplots(figsize=(figure_size, figure_size))

    if contour_filled:
        cf = ax.contourf(X, Y, values, levels=levels, cmap=cmap)
        fig.colorbar(cf, ax=ax, label=str(attrs["plot_label"]))

    CS = ax.contour(X, Y, values, levels=levels, colors=contour_color, linewidths=1.5)
    contour_labels = ax.clabel(CS, fontsize=CONTOUR_LABEL_FONTSIZE, inline=True, fmt="%.3g", colors=CONTOUR_LABEL_COLOR)
    _style_contour_labels(contour_labels)

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
    _apply_zoom_limits(ax, zoom_window)
    fig.tight_layout()

    output_path = output or default_output_name(
        slice_file,
        field_name,
        slice_tag,
        output_format,
        attrs,
        value_normalization=value_normalization_mode,
        normalize=normalize,
        output_tag=f"{slice_tag}_contour",
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
    value_normalization_mode,
    normalize,
    interpolate,
    zoom_window=None,
):
    """Render a separate contour plot for one saved slice using yt."""
    import yt

    saved = _apply_normalization(
        _load_saved_slice(slice_file, field_name, slice_tag),
        value_normalization_mode,
        normalize,
        print_stats=False,
    )
    dataset, yt_field, saved = build_yt_slice_dataset(slice_file, field_name, slice_tag, saved=saved)
    values = _prepare_plot_values(saved["values"])
    attrs = saved["attrs"]
    horizontal_coords = np.asarray(saved["coord_horizontal"], dtype=np.float64)
    vertical_coords = np.asarray(saved["coord_vertical"], dtype=np.float64)
    output_path = output or default_output_name(
        slice_file,
        field_name,
        slice_tag,
        output_format,
        attrs,
        value_normalization=value_normalization_mode,
        normalize=normalize,
        output_tag=f"{slice_tag}_contour",
    )
    data_min, data_max, zmin, zmax, limits_source = _resolve_saved_color_limits(saved, vmin=vmin, vmax=vmax)
    levels = _resolve_contour_levels(
        np.array([data_min, data_max], dtype=np.float64),
        vmin,
        vmax,
        contour_levels,
        contour_values,
    )
    print(f"Contour color scaling uses {limits_source}: min={data_min:.6g}, max={data_max:.6g}")
    grid_x, grid_y, contour_field = _build_contour_grid(
        horizontal_coords,
        vertical_coords,
        values,
        interpolate=interpolate,
    )

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
    _apply_image_interpolation(image, interpolate)
    axes = window_plot.axes
    image.set_visible(False)

    if contour_filled:
        axes.contourf(grid_x, grid_y, contour_field, levels=levels, cmap=cmap)
        window_plot.cb.set_label(str(attrs["plot_label"]))

    contour_set = axes.contour(grid_x, grid_y, contour_field, levels=levels, colors=contour_color, linewidths=1.5)
    contour_labels = axes.clabel(contour_set, fontsize=CONTOUR_LABEL_FONTSIZE, inline=True, fmt="%.3g", colors=CONTOUR_LABEL_COLOR)
    _style_contour_labels(contour_labels)
    _apply_zoom_limits(axes, zoom_window)
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
    value_normalization_mode,
    normalize,
    interpolate,
    zoom_window=None,
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
            value_normalization_mode,
            normalize,
            interpolate,
            zoom_window=zoom_window,
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
            value_normalization_mode,
            normalize,
            interpolate,
            zoom_window=zoom_window,
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
            value_normalization_mode,
            normalize,
            interpolate,
            zoom_window=zoom_window,
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
    value_normalization_mode,
    normalize,
    interpolate,
    contour_filled=False,
    output_paths=None,
    metadata_paths=None,
    zoom_window=None,
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
        saved = _apply_normalization(
            _load_saved_slice(slice_file, field_name, slice_tag),
            value_normalization_mode,
            normalize,
            print_stats=False,
        )
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

        grid_x, grid_y, contour_field = _build_contour_grid(
            horizontal_coords,
            vertical_coords,
            values,
            interpolate=interpolate,
        )
        prepared.append(
            {
                "slice_file": slice_file,
                "attrs": attrs,
                "saved": saved,
                "values": values,
                "stored_limits": _resolve_saved_color_limits(saved)[:2],
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
        global_min = min(float(item["stored_limits"][0]) for item in prepared)
        global_max = max(float(item["stored_limits"][1]) for item in prepared)
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
    base_min = float(base_item["stored_limits"][0])
    base_max = float(base_item["stored_limits"][1])
    zmin = float(base_min if vmin is None else vmin)
    zmax = float(base_max if vmax is None else vmax)
    print(f"Comparison contour scaling uses stored limits: min={base_min:.6g}, max={base_max:.6g}")

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
    _apply_image_interpolation(image, interpolate)
    image.set_visible(False)
    ax = window_plot.axes

    for index, item in enumerate(prepared):
        color = COMPARE_COLORS[index]
        item["compare_color"] = color
        contour_set = ax.contour(
            item["grid_x"],
            item["grid_y"],
            item["contour_field"],
            levels=levels,
            colors=color,
            linewidths=1.5,
        )
        if index == len(prepared) - 1:
            contour_labels = ax.clabel(
                contour_set,
                fontsize=CONTOUR_LABEL_FONTSIZE,
                inline=True,
                fmt="%.3g",
                colors=_comparison_contour_label_color(len(prepared)),
            )
            _style_contour_labels(contour_labels)
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
    _apply_zoom_limits(ax, zoom_window)

    if output_paths is None:
        output_paths = [
            default_comparison_output_name(
                item["slice_file"],
                field_name,
                slice_tag,
                output_format,
                item["attrs"],
                value_normalization=value_normalization_mode,
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
                value_normalization=value_normalization_mode,
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
    value_normalization_mode,
    normalize,
    interpolate,
):
    """Render one saved slice plane to disk and/or screen with optional yt interpolation."""
    import yt

    saved = _apply_normalization(
        _load_saved_slice(slice_file, field_name, slice_tag),
        value_normalization_mode,
        normalize,
        print_stats=True,
    )
    dataset, yt_field, saved = build_yt_slice_dataset(slice_file, field_name, slice_tag, saved=saved)
    values = _prepare_plot_values(saved["values"])

    attrs = saved["attrs"]
    output_path = output or default_output_name(
        slice_file,
        field_name,
        slice_tag,
        output_format,
        attrs,
        value_normalization=value_normalization_mode,
        normalize=normalize,
    )
    data_min, data_max, zmin, zmax, limits_source = _resolve_saved_color_limits(saved, vmin=vmin, vmax=vmax)
    print(f"Colorbar source: {limits_source}")
    print(f"Colorbar data min: {data_min:.6g}")
    print(f"Colorbar data max: {data_max:.6g}")
    print(f"Colorbar min used: {zmin:.6g}")
    print(f"Colorbar max used: {zmax:.6g}")

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
    _apply_image_interpolation(image, interpolate)
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
#     saved = _load_saved_slice(slice_file, field_name, slice_tag)
#     values = np.asarray(saved["values"], dtype=np.float64).copy()
#     values[np.abs(values) < 1.0e-12] = 0.0
#     values = np.round(values, decimals=10)
#     horizontal_coords = saved["coord_horizontal"]
#     vertical_coords = saved["coord_vertical"]
#     extent = _plane_extent_from_arrays(horizontal_coords, vertical_coords)
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
    saved = _load_saved_slice(slice_file, field_name, slice_tag)
    attrs = saved["attrs"]
    print(f"Selected field: {field_name}")
    print(f"Selected slice: {slice_tag}")
    print(f"Axis: {attrs['axis']}")
    print(f"Plane index: {int(attrs['plane_index'])}")
    print(f"Plane coordinate: {float(attrs['plane_coord']):.6g}")
    print(f"Horizontal axis: {attrs['horizontal_axis']}")
    print(f"Vertical axis: {attrs['vertical_axis']}")
    if "global_min" in attrs and "global_max" in attrs:
        print(f"Stored global 3D colorbar min: {float(attrs['global_min']):.6g}")
        print(f"Stored global 3D colorbar max: {float(attrs['global_max']):.6g}")
    if "global_rms" in attrs:
        print(f"Stored global 3D RMS normalization: {float(attrs['global_rms']):.6g}")
    print(f"Stored value normalization: {str(attrs.get('value_normalization', 'none'))}")
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
    print(f"Interpolation: {'enabled' if args.interpolate else 'disabled'}")

    if args.list:
        print_summary(slice_file)
        return

    if args.field is None and args.slice_tag is None:
        print_summary(slice_file)
        return

    _ensure_yt_imported()
    slice_tags = _resolve_slice_tags(slice_file, args.field, args.slice_tag)

    for index, slice_tag in enumerate(slice_tags):
        if index > 0:
            print()
        print_saved_slice_metadata(slice_file, args.field, slice_tag)

        if args.yt_info:
            print_yt_summary(slice_file, args.field, slice_tag)

        output_path = args.output if len(slice_tags) == 1 else None
        render_saved_slice(
            slice_file,
            args.field,
            slice_tag,
            args.cmap,
            args.width,
            args.vmin,
            args.vmax,
            output_path,
            args.format,
            args.plot,
            args.dpi,
            args.figsize,
            args.value_normalization,
            args.normalize,
            args.interpolate,
        )

        if args.render_contour:
            render_saved_slice_contour(
                slice_file,
                args.field,
                slice_tag,
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
                args.value_normalization,
                args.normalize,
                args.interpolate,
            )
            saved = _load_saved_slice(slice_file, args.field, slice_tag)
            zoom_output = default_zoom_contour_output_name(
                slice_file,
                args.field,
                slice_tag,
                args.format,
                saved["attrs"],
                value_normalization=args.value_normalization,
                normalize=args.normalize,
            )
            render_saved_slice_contour(
                slice_file,
                args.field,
                slice_tag,
                args.cmap,
                args.width,
                args.vmin,
                args.vmax,
                zoom_output,
                args.format,
                args.plot,
                args.dpi,
                args.figsize,
                args.contour_levels,
                args.contour_values,
                args.contour_filled,
                args.contour_color,
                args.contour_backend,
                args.value_normalization,
                args.normalize,
                args.interpolate,
                zoom_window=args.zoom_window,
            )


def main():
    parser = argparse.ArgumentParser(description="Inspect and replot saved 2D slice data from *_slices.h5 files.")
    parser.add_argument("slice_file", nargs="+", help="One or more paths to combined *_slices.h5 files")
    parser.add_argument("--list", action="store_true", help="Print the available fields and slices, then exit.")
    parser.add_argument(
        "--field",
        default=None,
        help="Field selector to replot: 1=vorticity_magnitude, 2=velocity_magnitude, 3=the saved scalar field when unique, or an explicit field name such as q_criterion, r_criterion, or density_gradient_magnitude.",
    )
    parser.add_argument(
        "--slice",
        dest="slice_tag",
        default=None,
        help="Optional slice selector to replot: 1=xy_center, 2=xy_face, 3=yz_face, 4=zx_face, or an explicit saved slice tag. If omitted, all saved slices for the selected field are rendered.",
    )
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
        "--value-normalization",
        default="saved",
        choices=["saved", "none", "global_rms"],
        help="Saved-value normalization to display on replot: 'saved' preserves the slice file's stored normalization, 'none' shows raw values, and 'global_rms' applies the stored full-volume RMS when available.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply an extra TGV-style normalization on top of the selected value normalization: vorticity uses U0/L and velocity uses U0.",
    )
    parser.add_argument(
        "--yt-info",
        action="store_true",
        help="Construct a one-cell-thick yt uniform-grid dataset for the selected slice and print its summary.",
    )
    parser.add_argument(
        "--contour",
        action="store_true",
        help="Also render contour plots alongside the default yt colormap plots. Matching zoomed contour plots are written automatically.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Deprecated. Comparison contour overlays now run automatically when more than one slice file is provided.",
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
        "--interpolate",
        default="1",
        choices=["0", "1"],
        help="Interpolation switch for both 2D slice images and contour plots: 1 keeps the current interpolated behavior, 0 disables interpolation and uses the raw saved grid.",
    )
    parser.add_argument(
        "--zoom-window",
        default=",".join(str(value) for value in DEFAULT_ZOOM_WINDOW),
        help="Zoom window for the extra zoom contour plots. Use xmin,xmax to apply the same range to both axes, or xmin,xmax,ymin,ymax. Default is 0.0,0.5.",
    )
    parser.add_argument(
        "--contour-backend",
        default="yt",
        choices=["auto", "yt", "matplotlib"],
        help="Contour renderer to use. Default is 'yt'. 'auto' tries yt first and falls back to matplotlib.",
    )
    args = parser.parse_args()
    if args.field is not None:
        args.field = _resolve_field_name(args.slice_file[0], args.field)
    if args.slice_tag is not None:
        args.slice_tag = _resolve_slice_name(args.slice_tag)
    args.zoom_window = _parse_zoom_window(args.zoom_window)
    args.interpolate = bool(int(args.interpolate))
    args.auto_compare = len(args.slice_file) > 1
    args.render_contour = args.contour or args.auto_compare

    if args.field is None and args.slice_tag is not None:
        raise SystemExit("--slice requires --field.")
    if args.output is not None and len(args.slice_file) > 1:
        raise SystemExit("--output can only be used when processing a single slice file.")
    if args.output is not None and args.slice_tag is None:
        raise SystemExit("--output requires --slice because all-slice mode writes one file per slice directory.")
    if args.auto_compare and not args.list and args.field is not None:
        if len(args.slice_file) > len(COMPARE_COLORS):
            raise SystemExit("Automatic comparison supports at most three slice files.")

    if not args.list and args.field is not None:
        args.slice_tags = _resolve_slice_tags(args.slice_file[0], args.field, args.slice_tag)
        if args.auto_compare:
            for slice_file in args.slice_file[1:]:
                current_tags = _resolve_slice_tags(slice_file, args.field, args.slice_tag)
                if current_tags != args.slice_tags:
                    raise SystemExit(
                        "All compared slice files must provide the same slice tags for the selected field. "
                        f"Expected {args.slice_tags} in {os.path.abspath(slice_file)}, got {current_tags}."
                    )
    else:
        args.slice_tags = []

    for index, slice_file in enumerate(args.slice_file):
        if index > 0:
            print()
            print("=" * 80)
            print()
        process_slice_file(slice_file, args)

    if args.auto_compare and not args.list and args.field is not None:
        for index, slice_tag in enumerate(args.slice_tags):
            print()
            print("=" * 80)
            print()
            if len(args.slice_tags) == 1:
                print("Rendering comparison contour overlay...")
            else:
                print(f"Rendering comparison contour overlay for {slice_tag}...")
            render_compared_contours(
                args.slice_file,
                args.field,
                slice_tag,
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
                args.value_normalization,
                args.normalize,
                args.interpolate,
                contour_filled=args.contour_filled,
            )
            zoom_output_paths = []
            zoom_metadata_paths = []
            for slice_file in args.slice_file:
                saved = _load_saved_slice(slice_file, args.field, slice_tag)
                zoom_output_paths.append(
                    default_zoom_comparison_output_name(
                        slice_file,
                        args.field,
                        slice_tag,
                        args.format,
                        saved["attrs"],
                        value_normalization=args.value_normalization,
                        normalize=args.normalize,
                    )
                )
                zoom_metadata_paths.append(
                    default_zoom_comparison_metadata_name(
                        slice_file,
                        args.field,
                        slice_tag,
                        saved["attrs"],
                        value_normalization=args.value_normalization,
                        normalize=args.normalize,
                    )
                )
            render_compared_contours(
                args.slice_file,
                args.field,
                slice_tag,
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
                args.value_normalization,
                args.normalize,
                args.interpolate,
                contour_filled=args.contour_filled,
                output_paths=zoom_output_paths,
                metadata_paths=zoom_metadata_paths,
                zoom_window=args.zoom_window,
            )


if __name__ == "__main__":
    main()
