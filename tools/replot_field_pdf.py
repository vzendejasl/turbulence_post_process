#!/usr/bin/env python3
"""Inspect, export, and replot stored full-field PDFs from a *_slices.h5 file.

Examples:
  List stored PDFs:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --list

  Print metadata for one stored PDF:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --metadata

  Replot the normalized dilatation PDF:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --pdf normalized_dilatation

  Replot by numeric selector shown in --list output:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --pdf 1

  Replot the normalized velocity, density, pressure, or Mach-number PDF:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --pdf normalized_velocity_magnitude
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --pdf normalized_density
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --pdf normalized_pressure
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --pdf normalized_mach_number

  Use the normalized convenience flag with a field shorthand:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --pdf density --normalized
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --pdf mach --normalized

  Replot on a logarithmic y-axis:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --y-scale log

  Replot the stored normalized PDF back in raw field units:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --x-normalization raw

  Restrict the x-axis range to [-20, 20]:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --x-range -20 20

  Export PDF data to CSV:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --export-csv dilatation_pdf.csv

  Save to a custom output path:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --output my_plot.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from postprocess_vis.pdfs import export_field_pdf_csv
from postprocess_vis.pdfs import FIELD_PDF_REGISTRY
from postprocess_vis.pdfs import field_pdf_output_path
from postprocess_vis.pdfs import field_pdf_metadata_text
from postprocess_vis.pdfs import plot_field_pdf
from postprocess_vis.pdfs import rescale_field_pdf_for_plot
from postprocess_vis.slice_data import list_available_pdfs
from postprocess_vis.slice_data import load_saved_pdf


PDF_NAME_ALIASES = {
    "normalized_dilation": "normalized_dilatation",
    "dilation": "normalized_dilatation",
    "dilatation": "normalized_dilatation",
    "div_u": "normalized_dilatation",
    "velocity": "normalized_velocity_magnitude",
    "normalized_velocity": "normalized_velocity_magnitude",
    "velocity_magnitude": "normalized_velocity_magnitude",
    "density": "normalized_density",
    "pressure": "normalized_pressure",
    "mach": "normalized_mach_number",
    "normalized_mach": "normalized_mach_number",
    "mach_number": "normalized_mach_number",
}


def ordered_available_pdf_names(pdfs):
    """Return available stored PDF names in a stable, user-facing order."""
    ordered = [name for name in FIELD_PDF_REGISTRY if name in pdfs]
    ordered.extend(sorted(name for name in pdfs if name not in FIELD_PDF_REGISTRY))
    return ordered


def resolve_pdf_selector(selector, pdfs, *, normalized=False):
    """Resolve one CLI selector into a stored PDF name."""
    available = ordered_available_pdf_names(pdfs)
    if selector is None:
        return None

    token = str(selector).strip().lower()
    if not token:
        raise ValueError("Empty PDF selector.")

    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(available):
            return available[index - 1]
        raise ValueError(
            f"PDF selector {index} is out of range. Available selectors: "
            + ", ".join(f"{idx + 1}={name}" for idx, name in enumerate(available))
        )

    if normalized and not token.startswith("normalized_"):
        token = f"normalized_{token}"
    token = PDF_NAME_ALIASES.get(token, token)
    if token in pdfs:
        return token

    raise ValueError(
        f"Unknown PDF selector '{selector}'. Available selectors: "
        + ", ".join(f"{idx + 1}={name}" for idx, name in enumerate(available))
    )


def print_summary(filepath):
    """Print the stored field PDFs available in one slice-data file."""
    pdfs = list_available_pdfs(filepath)
    print(f"Slice file: {os.path.abspath(filepath)}")
    if not pdfs:
        print("Stored PDFs: none")
        return
    print("Stored PDFs:")
    for index, pdf_name in enumerate(ordered_available_pdf_names(pdfs), start=1):
        summary = pdfs[pdf_name]
        print(
            f"  {index}. {pdf_name}: source_field={summary['source_field']}, "
            f"normalization={summary['normalization']}, bins={summary['bin_count']}"
        )


def print_pdf_metadata(filepath, pdf_name):
    """Print the stored metadata for one field PDF."""
    loaded = load_saved_pdf(filepath, pdf_name)
    pdf_result = dict(loaded["attrs"])
    pdf_result["bin_edges"] = loaded["bin_edges"]
    pdf_result["bin_centers"] = loaded["bin_centers"]
    pdf_result["counts"] = loaded["counts"]
    pdf_result["pdf"] = loaded["pdf"]
    print(field_pdf_metadata_text(pdf_result), end="")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and replot stored full-field PDFs from a *_slices.h5 file."
    )
    parser.add_argument("slice_file", help="Path to a combined *_slices.h5 file")
    parser.add_argument("--list", action="store_true", help="Print the available stored PDFs, then exit.")
    parser.add_argument(
        "--pdf",
        default=None,
        help=(
            "Stored PDF selector to replot or export. Accepts a PDF name or "
            "a number from --list output, e.g. 1, normalized_dilatation, density, mach."
        ),
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Interpret short field names passed to --pdf as normalized PDF names, e.g. --pdf density --normalized or --pdf mach --normalized.",
    )
    parser.add_argument("--output", "--out", dest="output", default=None, help="Optional output plot path")
    parser.add_argument("--format", "--fmt", dest="format", default="pdf", choices=["pdf", "png"], help="Output image format when --output is omitted.")
    parser.add_argument("--plot", action="store_true", help="Also display the plot after saving")
    parser.add_argument("--metadata", "--meta", dest="metadata", action="store_true", help="Print the stored metadata for the selected PDF.")
    parser.add_argument("--export-csv", "--csv", dest="export_csv", default=None, help="Optional CSV export path for the selected PDF data.")
    parser.add_argument(
        "--y-scale",
        "--yscale",
        dest="y_scale",
        default="log",
        choices=["linear", "log"],
        help="Vertical scale for the plotted PDF. Default is 'log'.",
    )
    parser.add_argument(
        "--x-normalization",
        "--x-norm",
        dest="x_normalization",
        default="stored",
        choices=["stored", "raw"],
        help="How to plot the x-axis. 'stored' uses the saved PDF variable, 'raw' rescales back to the original field units when the stored normalization scale is available.",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional x-axis display range, e.g. --x-range -20 20.",
    )
    args = parser.parse_args()

    if args.list or args.pdf is None:
        print_summary(args.slice_file)
        if args.pdf is None:
            return

    available_pdfs = list_available_pdfs(args.slice_file)
    try:
        resolved_pdf_name = resolve_pdf_selector(
            args.pdf,
            available_pdfs,
            normalized=args.normalized,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    loaded = load_saved_pdf(args.slice_file, resolved_pdf_name)
    pdf_result = dict(loaded["attrs"])
    pdf_result["bin_edges"] = loaded["bin_edges"]
    pdf_result["bin_centers"] = loaded["bin_centers"]
    pdf_result["counts"] = loaded["counts"]
    pdf_result["pdf"] = loaded["pdf"]

    # Override stored labels with current registry values so old HDF5 files
    # pick up label changes without needing to be recomputed.
    if resolved_pdf_name in FIELD_PDF_REGISTRY:
        for key in ("x_label", "y_label", "plot_title"):
            if key in FIELD_PDF_REGISTRY[resolved_pdf_name]:
                pdf_result[key] = FIELD_PDF_REGISTRY[resolved_pdf_name][key]

    if args.metadata:
        print(field_pdf_metadata_text(pdf_result), end="")

    if args.export_csv:
        export_field_pdf_csv(pdf_result, args.export_csv)
        print(f"Saved CSV: {os.path.abspath(args.export_csv)}")

    output_anchor = str(pdf_result.get("source_h5", args.slice_file))
    plotted_pdf_result = rescale_field_pdf_for_plot(pdf_result, x_normalization=args.x_normalization)
    output_path = args.output or field_pdf_output_path(output_anchor, resolved_pdf_name, output_format=args.format)
    if args.y_scale != "linear" and not args.output:
        stem, ext = os.path.splitext(output_path)
        output_path = f"{stem}_{args.y_scale}_scale{ext}"
    plot_field_pdf(
        pdf_result,
        output_path,
        plot=args.plot,
        y_scale=args.y_scale,
        x_normalization=args.x_normalization,
        x_range=args.x_range,
        backend="yt",
    )
    print(
        "X range for plot: "
        f"[{float(plotted_pdf_result['bin_edges'][0]):.6g}, {float(plotted_pdf_result['bin_edges'][-1]):.6g}] "
        f"({args.x_normalization} units)"
    )
    if args.x_normalization != "stored":
        print(
            "Stored normalized x range: "
            f"[{float(pdf_result['bin_edges'][0]):.6g}, {float(pdf_result['bin_edges'][-1]):.6g}]"
        )
    print(f"Saved field PDF plot: {output_path}")


if __name__ == "__main__":
    main()
