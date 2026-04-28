#!/usr/bin/env python3
"""Inspect, export, and replot stored full-field PDFs from a *_slices.h5 file.

Examples:
  List stored PDFs:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --list

  Print metadata for one stored PDF:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --metadata

  Replot on a logarithmic y-axis:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --y-scale log

  Replot the stored normalized PDF back in raw field units:
    python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
      --pdf normalized_dilatation \
      --x-normalization raw
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
from postprocess_vis.pdfs import field_pdf_output_path
from postprocess_vis.pdfs import field_pdf_metadata_text
from postprocess_vis.pdfs import plot_field_pdf
from postprocess_vis.pdfs import rescale_field_pdf_for_plot
from postprocess_vis.slice_data import list_available_pdfs
from postprocess_vis.slice_data import load_saved_pdf


def print_summary(filepath):
    """Print the stored field PDFs available in one slice-data file."""
    pdfs = list_available_pdfs(filepath)
    print(f"Slice file: {os.path.abspath(filepath)}")
    if not pdfs:
        print("Stored PDFs: none")
        return
    print("Stored PDFs:")
    for pdf_name, summary in pdfs.items():
        print(
            f"  {pdf_name}: source_field={summary['source_field']}, "
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
    parser.add_argument("--pdf", default=None, help="Stored PDF name to replot or export, e.g. normalized_dilatation.")
    parser.add_argument("--output", "--out", dest="output", default=None, help="Optional output plot path")
    parser.add_argument("--format", "--fmt", dest="format", default="pdf", choices=["pdf", "png"], help="Output image format when --output is omitted.")
    parser.add_argument("--plot", action="store_true", help="Also display the plot after saving")
    parser.add_argument("--metadata", "--meta", dest="metadata", action="store_true", help="Print the stored metadata for the selected PDF.")
    parser.add_argument("--export-csv", "--csv", dest="export_csv", default=None, help="Optional CSV export path for the selected PDF data.")
    parser.add_argument(
        "--y-scale",
        "--yscale",
        dest="y_scale",
        default="linear",
        choices=["linear", "log"],
        help="Vertical scale for the plotted PDF. 'log' keeps the same PDF values but displays the y-axis logarithmically.",
    )
    parser.add_argument(
        "--x-normalization",
        "--x-norm",
        dest="x_normalization",
        default="stored",
        choices=["stored", "raw"],
        help="How to plot the x-axis. 'stored' uses the saved PDF variable, 'raw' rescales back to the original field units when the stored normalization scale is available.",
    )
    args = parser.parse_args()

    if args.list or args.pdf is None:
        print_summary(args.slice_file)
        if args.pdf is None:
            return

    loaded = load_saved_pdf(args.slice_file, args.pdf)
    pdf_result = dict(loaded["attrs"])
    pdf_result["bin_edges"] = loaded["bin_edges"]
    pdf_result["bin_centers"] = loaded["bin_centers"]
    pdf_result["counts"] = loaded["counts"]
    pdf_result["pdf"] = loaded["pdf"]

    if args.metadata:
        print(field_pdf_metadata_text(pdf_result), end="")

    if args.export_csv:
        export_field_pdf_csv(pdf_result, args.export_csv)
        print(f"Saved CSV: {os.path.abspath(args.export_csv)}")

    output_anchor = str(pdf_result.get("source_h5", args.slice_file))
    output_path = args.output or field_pdf_output_path(output_anchor, args.pdf, output_format=args.format)
    plotted_pdf_result = rescale_field_pdf_for_plot(pdf_result, x_normalization=args.x_normalization)
    plot_field_pdf(
        pdf_result,
        output_path,
        plot=args.plot,
        y_scale=args.y_scale,
        x_normalization=args.x_normalization,
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
