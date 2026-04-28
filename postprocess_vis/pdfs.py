"""Helpers for distributed field PDFs stored alongside slice data."""

from __future__ import annotations

import csv
import os
import warnings
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator

import numpy as np


DEFAULT_FIELD_PDF_BINS = 256


def _trimmed_decimal_label(value):
    """Format one tick with up to two decimals while keeping one decimal for integers."""
    formatted = f"{float(value):.2f}".rstrip("0")
    if formatted.endswith("."):
        formatted += "0"
    return formatted


FIELD_PDF_REGISTRY = {
    "normalized_dilatation": {
        "pdf_name": "normalized_dilatation",
        "source_field": "div_u",
        "source_field_family": "divergence",
        "normalization": "global_std",
        "plot_title": "Normalized Dilatation PDF",
        "x_label": r"$\left(\theta - \langle \theta \rangle\right) / \mathrm{std}(\theta)$",
        "raw_x_label": r"$\theta$",
        "y_label": "PDF",
    },
    "normalized_velocity_magnitude": {
        "pdf_name": "normalized_velocity_magnitude",
        "source_field": "velocity_magnitude",
        "source_field_family": "velocity",
        "normalization": "global_std",
        "plot_title": "Normalized Velocity Magnitude PDF",
        "x_label": r"$\left(|\mathbf{u}| - \langle |\mathbf{u}| \rangle\right) / \mathrm{std}(|\mathbf{u}|)$",
        "raw_x_label": r"$|\mathbf{u}|$",
        "y_label": "PDF",
    },
    "normalized_density": {
        "pdf_name": "normalized_density",
        "source_field": "density",
        "source_field_family": "scalar",
        "normalization": "global_std",
        "plot_title": "Normalized Density PDF",
        "x_label": r"$\left(\rho - \langle \rho \rangle\right) / \mathrm{std}(\rho)$",
        "raw_x_label": r"$\rho$",
        "y_label": "PDF",
    },
    "normalized_pressure": {
        "pdf_name": "normalized_pressure",
        "source_field": "pressure",
        "source_field_family": "scalar",
        "normalization": "global_std",
        "plot_title": "Normalized Pressure PDF",
        "x_label": r"$\left(p - \langle p \rangle\right) / \mathrm{std}(p)$",
        "raw_x_label": r"$p$",
        "y_label": "PDF",
    },
    "normalized_mach_number": {
        "pdf_name": "normalized_mach_number",
        "source_field": "mach_number",
        "source_field_family": "thermo",
        "normalization": "global_std",
        "plot_title": "Normalized Mach Number PDF",
        "x_label": r"$\left(M - \langle M \rangle\right) / \mathrm{std}(M)$",
        "raw_x_label": r"$M$",
        "y_label": "PDF",
    },
}


def default_field_pdf_specs(field_specs, force_normalized_dilatation=False):
    """Return the default full-field PDF specs for one visualization run."""
    available_fields = {field_label for _, field_label, _, _ in field_specs}
    specs = []
    if force_normalized_dilatation or "div_u" in available_fields:
        specs.append(dict(FIELD_PDF_REGISTRY["normalized_dilatation"]))
    if "velocity_magnitude" in available_fields:
        specs.append(dict(FIELD_PDF_REGISTRY["normalized_velocity_magnitude"]))
    if "density" in available_fields:
        specs.append(dict(FIELD_PDF_REGISTRY["normalized_density"]))
    if "pressure" in available_fields:
        specs.append(dict(FIELD_PDF_REGISTRY["normalized_pressure"]))
    if "mach_number" in available_fields:
        specs.append(dict(FIELD_PDF_REGISTRY["normalized_mach_number"]))
    return specs


def _comm_allreduce(value, comm, op_name):
    """Apply one named MPI reduction without importing MPI at module import time."""
    from mpi4py import MPI

    return comm.allreduce(value, op=getattr(MPI, op_name))


def _resolved_value_range(local_values, comm, value_range=None):
    """Return one finite histogram range for a distributed array."""
    local_values = np.asarray(local_values, dtype=np.float64)
    if value_range is None:
        if local_values.size == 0:
            local_min = np.inf
            local_max = -np.inf
        else:
            local_min = float(np.min(local_values))
            local_max = float(np.max(local_values))
        global_min = float(_comm_allreduce(local_min, comm, "MIN"))
        global_max = float(_comm_allreduce(local_max, comm, "MAX"))
    else:
        global_min = float(value_range[0])
        global_max = float(value_range[1])

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("Cannot build a PDF from non-finite global range values.")

    if np.isclose(global_min, global_max):
        delta = max(abs(global_min) * 1.0e-6, 1.0e-12)
        global_min -= delta
        global_max += delta

    return float(global_min), float(global_max)


def compute_distributed_field_pdf(
    local_values,
    comm,
    *,
    bins=DEFAULT_FIELD_PDF_BINS,
    value_range=None,
    normalization_scale=1.0,
    normalization_offset=0.0,
    pdf_name,
    source_field,
    normalization="none",
    plot_title="Field PDF",
    x_label="Value",
    raw_x_label=None,
    y_label="PDF",
):
    """Return a distributed 1D PDF for one field stored across MPI ranks."""
    local_values = np.asarray(local_values, dtype=np.float64)
    if local_values.ndim == 0:
        local_values = local_values.reshape(1)

    if bins <= 0:
        raise ValueError("The number of PDF bins must be positive.")

    scale = float(normalization_scale)
    if not np.isfinite(scale) or abs(scale) <= 1.0e-30:
        raise ValueError(
            f"Cannot compute PDF '{pdf_name}' because normalization_scale={scale!r} is not usable."
        )
    offset = float(normalization_offset)
    if not np.isfinite(offset):
        raise ValueError(
            f"Cannot compute PDF '{pdf_name}' because normalization_offset={offset!r} is not usable."
        )

    normalized_local_values = (local_values - offset) / scale
    total_samples = int(_comm_allreduce(int(normalized_local_values.size), comm, "SUM"))
    if total_samples <= 0:
        raise ValueError(f"Cannot compute PDF '{pdf_name}' because there are no samples.")

    range_min, range_max = _resolved_value_range(normalized_local_values, comm, value_range=value_range)
    bin_edges = np.linspace(range_min, range_max, int(bins) + 1, dtype=np.float64)
    local_hist, _ = np.histogram(normalized_local_values, bins=bin_edges)
    global_hist = np.asarray(_comm_allreduce(local_hist.astype(np.int64), comm, "SUM"), dtype=np.int64)

    local_underflow = int(np.count_nonzero(normalized_local_values < bin_edges[0]))
    local_overflow = int(np.count_nonzero(normalized_local_values > bin_edges[-1]))
    global_underflow = int(_comm_allreduce(local_underflow, comm, "SUM"))
    global_overflow = int(_comm_allreduce(local_overflow, comm, "SUM"))

    counts_total = int(np.sum(global_hist, dtype=np.int64))
    if counts_total != total_samples - global_underflow - global_overflow:
        raise ValueError(
            f"Global histogram counts ({counts_total}) do not match the expected in-range sample count "
            f"({total_samples - global_underflow - global_overflow}) for PDF '{pdf_name}'."
        )

    bin_widths = np.diff(bin_edges)
    pdf = np.asarray(global_hist, dtype=np.float64) / (float(total_samples) * bin_widths)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf_integral = float(np.sum(pdf * bin_widths, dtype=np.float64))

    binning_warning = (
        "This PDF uses a variable per-run bin range built from the GLOBAL min/max of the normalized field. "
        "It is suitable for single-run inspection, but it is NOT directly comparable across different runs until "
        "those PDFs are rebinned or replotted onto a shared fixed x-range."
    )
    range_warning = ""
    if global_underflow or global_overflow:
        range_warning = (
            f"Detected {global_underflow} underflow and {global_overflow} overflow sample(s) outside the stored "
            "PDF range. Widen the range if you need the clipped tails."
        )

    result = {
        "pdf_name": str(pdf_name),
        "source_field": str(source_field),
        "normalization": str(normalization),
        "plot_title": str(plot_title),
        "x_label": str(x_label),
        "y_label": str(y_label),
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "counts": global_hist,
        "pdf": pdf,
        "total_samples": int(total_samples),
        "in_range_samples": int(counts_total),
        "underflow_count": int(global_underflow),
        "overflow_count": int(global_overflow),
        "normalization_scale": float(scale),
        "normalization_offset": float(offset),
        "value_range_min": float(bin_edges[0]),
        "value_range_max": float(bin_edges[-1]),
        "bin_count": int(bins),
        "range_mode": "variable" if value_range is None else "fixed",
        "pdf_integral": float(pdf_integral),
        "binning_warning": binning_warning,
        "range_warning": range_warning,
    }
    if raw_x_label is not None:
        result["raw_x_label"] = str(raw_x_label)
    return result


def field_pdf_output_path(source_path, pdf_name, output_format="pdf"):
    """Return the default plot path for one stored field PDF."""
    source_path = os.path.abspath(source_path)
    directory = os.path.dirname(source_path)
    output_dir = os.path.join(directory, "slice_plots", "pdfs")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(output_dir, f"{pdf_name}_{base}.{output_format}")


def field_pdf_metadata_path(output_path):
    """Return the metadata path for one field-PDF plot."""
    stem, _ = os.path.splitext(output_path)
    return f"{stem}_metadata.txt"


def field_pdf_metadata_text(pdf_result):
    """Return metadata text for one stored field PDF."""
    lines = [
        f"PDF name: {pdf_result['pdf_name']}",
        f"Source field: {pdf_result['source_field']}",
        f"Source field min: {float(pdf_result.get('source_field_min', 0.0)):.16e}",
        f"Source field max: {float(pdf_result.get('source_field_max', 0.0)):.16e}",
        f"Source field mean: {float(pdf_result.get('source_field_mean', 0.0)):.16e}",
        f"Source field std: {float(pdf_result.get('source_field_std', pdf_result.get('normalization_scale', 0.0))):.16e}",
        f"Normalization: {pdf_result['normalization']}",
        f"Normalization scale: {float(pdf_result['normalization_scale']):.16e}",
        f"Normalization offset: {float(pdf_result.get('normalization_offset', 0.0)):.16e}",
        f"Total samples: {int(pdf_result['total_samples'])}",
        f"In-range samples: {int(pdf_result['in_range_samples'])}",
        f"Underflow count: {int(pdf_result['underflow_count'])}",
        f"Overflow count: {int(pdf_result['overflow_count'])}",
        f"Bin count: {int(pdf_result['bin_count'])}",
        f"Range mode: {pdf_result['range_mode']}",
        f"Stored range min: {float(pdf_result['value_range_min']):.16e}",
        f"Stored range max: {float(pdf_result['value_range_max']):.16e}",
        f"PDF integral: {float(pdf_result['pdf_integral']):.16e}",
        f"Binning warning: {pdf_result['binning_warning']}",
    ]
    if "measured_normalization_scale" in pdf_result:
        lines.append(
            f"Measured normalization scale: {float(pdf_result['measured_normalization_scale']):.16e}"
        )
    if "near_zero_field_treated_as_zero" in pdf_result:
        lines.append(
            f"Near-zero field treated as zero: {bool(pdf_result['near_zero_field_treated_as_zero'])}"
        )
    if str(pdf_result.get("range_warning", "")).strip():
        lines.append(f"Range warning: {pdf_result['range_warning']}")
    return "\n".join(lines) + "\n"


def write_field_pdf_metadata(output_path, pdf_result):
    """Write metadata text for one field-PDF plot."""
    metadata_path = field_pdf_metadata_path(output_path)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        handle.write(field_pdf_metadata_text(pdf_result))
    return metadata_path


def rescale_field_pdf_for_plot(pdf_result, x_normalization="stored"):
    """Return a plotting copy of one stored PDF result with optional x-axis rescaling."""
    resolved = str(x_normalization or "stored").strip().lower()
    if resolved not in {"stored", "raw"}:
        raise ValueError(f"Unsupported x_normalization '{x_normalization}'. Use one of: stored, raw.")

    result = dict(pdf_result)
    result["bin_edges"] = np.asarray(pdf_result["bin_edges"], dtype=np.float64).copy()
    result["bin_centers"] = np.asarray(pdf_result["bin_centers"], dtype=np.float64).copy()
    result["counts"] = np.asarray(pdf_result["counts"])
    result["pdf"] = np.asarray(pdf_result["pdf"], dtype=np.float64).copy()

    if resolved == "stored":
        return result

    scale = float(pdf_result.get("measured_normalization_scale", pdf_result.get("normalization_scale", 0.0)))
    offset = float(pdf_result.get("normalization_offset", 0.0))
    if not np.isfinite(scale) or abs(scale) <= 1.0e-30:
        raise ValueError(
            "Cannot replot this stored PDF in raw units because the normalization scale is not usable."
        )

    result["bin_edges"] = result["bin_edges"] * scale + offset
    result["bin_centers"] = result["bin_centers"] * scale + offset
    result["pdf"] = result["pdf"] / abs(scale)
    result["value_range_min"] = float(result["bin_edges"][0])
    result["value_range_max"] = float(result["bin_edges"][-1])
    result["x_label"] = str(pdf_result.get("raw_x_label", pdf_result.get("source_field", "Value")))
    result["plot_title"] = f"{pdf_result.get('plot_title', 'Field PDF')} [raw units]"
    result["x_normalization"] = "raw"
    return result


def _configure_pdf_axes(ax, *, x_formatter, y_scale):
    """Apply shared PDF-axis tick density/formatting."""
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if str(y_scale or "linear").strip().lower() == "log":
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _trimmed_decimal_label(value)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _plot_field_pdf_yt(plotted, output_path, *, plot=False, y_scale="log", x_range=None):
    """Plot one stored field PDF using a true yt LinePlot render."""
    import yt
    import matplotlib.pyplot as plt
    from yt.funcs import mylog

    centers = np.asarray(plotted["bin_centers"], dtype=np.float64)
    pdf_values = np.asarray(plotted["pdf"], dtype=np.float64)
    bin_edges = np.asarray(plotted["bin_edges"], dtype=np.float64)
    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])

    # Build a uniformly sampled 1D field over the PDF x-range and let yt
    # render the line directly through LinePlot instead of redrawing it with
    # matplotlib on cleared axes.
    line_sample_count = max(int(centers.size), 1000)
    sample_x = centers
    sample_pdf = pdf_values.reshape(int(centers.size), 1, 1)
    bbox = np.array([[x_min, x_max], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    previous_log_level = mylog.level
    mylog.setLevel("ERROR")
    try:
        ds = yt.load_uniform_grid(
            {"pdf": sample_pdf},
            sample_pdf.shape,
            bbox=bbox,
            nprocs=1,
            periodicity=(False, False, False),
            length_unit=1.0,
        )
        field = ("stream", "pdf")
        lp = yt.LinePlot(
            ds,
            [field],
            (float(sample_x[0]), 0.5, 0.5),
            (float(sample_x[-1]), 0.5, 0.5),
            line_sample_count,
        )
        if str(y_scale or "linear").strip().lower() == "log":
            lp.set_log(field, True)
        else:
            lp.set_log(field, False)
        lp._setup_plots()

        plot_mpl = lp.plots[field]
        ax = plot_mpl.axes
        fig = plot_mpl.figure
        distance_span = float(sample_x[-1] - sample_x[0])

        ax.set_xlabel(str(plotted.get("x_label", "Value")))
        ax.set_ylabel(str(plotted.get("y_label", "PDF")))
        ax.set_title("")
        if x_range is not None:
            ax.set_xlim(float(x_range[0] - x_min), float(x_range[1] - x_min))
        else:
            ax.set_xlim(0.0, distance_span)
        _configure_pdf_axes(
            ax,
            x_formatter=FuncFormatter(lambda value, _: _trimmed_decimal_label(value + x_min)),
            y_scale=y_scale,
        )
        ax.grid(True, alpha=0.25)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
                category=UserWarning,
            )
            fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        if plot:
            plt.show()
        plt.close(fig)
    finally:
        mylog.setLevel(previous_log_level)
    return output_path


def _plot_field_pdf_matplotlib(plotted, output_path, *, plot=False, y_scale="log", x_range=None):
    """Fallback matplotlib renderer for field PDFs."""
    import matplotlib.pyplot as plt

    centers = np.asarray(plotted["bin_centers"], dtype=np.float64)
    pdf = np.asarray(plotted["pdf"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(centers, pdf, color="black", linewidth=1.5)
    ax.set_xlabel(str(plotted.get("x_label", "Value")))
    ax.set_ylabel(str(plotted.get("y_label", "PDF")))
    ax.set_title("")
    if x_range is not None:
        ax.set_xlim(float(x_range[0]), float(x_range[1]))
    if str(y_scale or "linear").strip().lower() == "log":
        ax.set_yscale("log")
    _configure_pdf_axes(
        ax,
        x_formatter=FuncFormatter(lambda value, _: _trimmed_decimal_label(value)),
        y_scale=y_scale,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    if plot:
        plt.show()
    plt.close(fig)
    return output_path


def plot_field_pdf(
    pdf_result,
    output_path,
    *,
    plot=False,
    y_scale="log",
    x_normalization="stored",
    x_range=None,
    backend="yt",
):
    """Plot one stored field PDF to disk using the requested rendering backend."""
    plotted = rescale_field_pdf_for_plot(pdf_result, x_normalization=x_normalization)
    resolved_y_scale = str(y_scale or "linear").strip().lower()
    if resolved_y_scale not in {"linear", "log"}:
        raise ValueError(f"Unsupported y_scale '{y_scale}'. Use one of: linear, log.")
    resolved_backend = str(backend or "yt").strip().lower()
    if resolved_backend == "yt":
        return _plot_field_pdf_yt(plotted, output_path, plot=plot, y_scale=resolved_y_scale, x_range=x_range)
    if resolved_backend == "matplotlib":
        return _plot_field_pdf_matplotlib(plotted, output_path, plot=plot, y_scale=resolved_y_scale, x_range=x_range)
    raise ValueError(f"Unsupported PDF plotting backend '{backend}'. Use one of: yt, matplotlib.")


def export_field_pdf_csv(pdf_result, output_path):
    """Export one stored field PDF to CSV."""
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", "bin_center", "count", "pdf"])
        for left, right, center, count, density in zip(
            np.asarray(pdf_result["bin_edges"][:-1], dtype=np.float64),
            np.asarray(pdf_result["bin_edges"][1:], dtype=np.float64),
            np.asarray(pdf_result["bin_centers"], dtype=np.float64),
            np.asarray(pdf_result["counts"], dtype=np.int64),
            np.asarray(pdf_result["pdf"], dtype=np.float64),
        ):
            writer.writerow([float(left), float(right), float(center), int(count), float(density)])
    return output_path


def print_field_pdf_summary(pdf_result, *, output_path=None):
    """Print a concise rank-0 summary for one field PDF."""
    print(f"PDF saved: {pdf_result['pdf_name']}")
    print(f"  Source field      : {pdf_result['source_field']}")
    print(f"  Source field min  : {float(pdf_result.get('source_field_min', 0.0)):.6g}")
    print(f"  Source field max  : {float(pdf_result.get('source_field_max', 0.0)):.6g}")
    print(f"  Source field mean : {float(pdf_result.get('source_field_mean', 0.0)):.6g}")
    print(
        "  Source field std  : "
        f"{float(pdf_result.get('source_field_std', pdf_result.get('normalization_scale', 0.0))):.6g}"
    )
    print(f"  Normalization     : {pdf_result['normalization']}")
    print(f"  Normalization scale: {float(pdf_result['normalization_scale']):.6g}")
    print(f"  Normalization offset: {float(pdf_result.get('normalization_offset', 0.0)):.6g}")
    if "measured_normalization_scale" in pdf_result:
        print(
            "  Measured normalization scale: "
            f"{float(pdf_result['measured_normalization_scale']):.6g}"
        )
    print(
        f"  Range             : [{float(pdf_result['value_range_min']):.6g}, "
        f"{float(pdf_result['value_range_max']):.6g}]"
    )
    print(f"  Bin count         : {int(pdf_result['bin_count'])}")
    print(f"  Total samples     : {int(pdf_result['total_samples'])}")
    print(f"  PDF integral      : {float(pdf_result['pdf_integral']):.6g}")
    print()
    print(f"  Warning           : {pdf_result['binning_warning']}")
    if str(pdf_result.get("range_warning", "")).strip():
        print(f"  Range warning     : {pdf_result['range_warning']}")
    if output_path:
        print()
        print(f"  Plot output       : {output_path}")
    print()


def serializable_field_pdf(pdf_result):
    """Return one field-PDF result with NumPy arrays preserved but attrs normalized."""
    normalized = dict(pdf_result)
    for key in ("bin_edges", "bin_centers", "counts", "pdf"):
        normalized[key] = np.asarray(pdf_result[key])
    return normalized
