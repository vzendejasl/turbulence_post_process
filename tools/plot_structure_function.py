#!/usr/bin/env python3
"""Plot saved second- or third-order longitudinal structure function files to PDF.

Examples:
  python tools/plot_structure_function.py -q2 path/to/foo_spectra_structure_function.txt
  python tools/plot_structure_function.py -q3 path/to/foo_spectra_structure_function_third_order.txt
  python tools/plot_structure_function.py -q2 path/to/foo_spectra_structure_function.txt --output foo_structure_function.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


def _default_output_path(input_path: Path) -> Path:
    if input_path.stem.endswith("_structure_function_third_order"):
        return input_path.with_name(f"{input_path.stem}_plot.pdf")
    if input_path.stem.endswith("_structure_function"):
        return input_path.with_name(f"{input_path.stem}_plot.pdf")
    return input_path.with_suffix(".pdf")


def _plot_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }


def _reference_plateau_q2(r_values: np.ndarray, compensated_values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    positive_mask = (r_values > 0.0) & (compensated_values > 0.0)
    if not np.any(positive_mask):
        return None, None

    r_positive = np.asarray(r_values[positive_mask], dtype=np.float64)
    comp_positive = np.asarray(compensated_values[positive_mask], dtype=np.float64)
    anchor_index = len(comp_positive) // 2
    anchor_value = 1.1 * float(comp_positive[anchor_index])
    if anchor_value <= 0.0:
        return None, None

    return r_positive, np.full_like(r_positive, anchor_value, dtype=np.float64)


def _reference_plateau_q3(r_values: np.ndarray, compensated_values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    positive_mask = (r_values > 0.0) & np.isfinite(compensated_values)
    if not np.any(positive_mask):
        return None, None

    r_positive = np.asarray(r_values[positive_mask], dtype=np.float64)
    comp_positive = np.asarray(compensated_values[positive_mask], dtype=np.float64)
    anchor_index = len(comp_positive) // 2
    anchor_value = float(comp_positive[anchor_index])
    return r_positive, np.full_like(r_positive, anchor_value, dtype=np.float64)


def _infer_order_from_path(input_path: Path) -> str:
    if input_path.stem.endswith("_structure_function_third_order"):
        return "q3"
    return "q2"


def _plot_q2(input_path: Path, output_path: Path) -> None:
    data = np.loadtxt(input_path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise SystemExit(f"Expected at least two columns in {input_path}")

    r_values = np.asarray(data[:, 0], dtype=np.float64)
    s_values = np.asarray(data[:, 1], dtype=np.float64)
    compensated_values = np.zeros_like(s_values, dtype=np.float64)
    positive_r_mask = r_values > 0.0
    compensated_values[positive_r_mask] = (
        s_values[positive_r_mask] * (r_values[positive_r_mask] ** (-2.0 / 3.0))
    )
    positive_mask = (r_values > 0.0) & (compensated_values > 0.0)
    if not np.any(positive_mask):
        raise SystemExit("Need at least one positive (r, r^(-2/3) S_L) pair for a log-log plot.")

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        ax.loglog(
            r_values[positive_mask],
            compensated_values[positive_mask],
            linewidth=2.0,
            label=r"$r^{-2/3} S_L(r)$",
        )
        r_ref, plateau = _reference_plateau_q2(r_values, compensated_values)
        if r_ref is not None and plateau is not None:
            ax.loglog(
                r_ref,
                plateau,
                linestyle="--",
                linewidth=1.5,
                label=r"$\propto r^{2/3}$ reference",
            )

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$r^{-2/3} S_L(r)$")
        ax.set_title("Compensated Longitudinal Second-Order Structure Function")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved structure-function plot: {output_path}")
    print("Order: q2")
    print("Plot scale: log-log")
    print("Plotted quantity: r^(-2/3) * S_L(r)")
    print("Reference guide: horizontal line anchored to the mid-range compensated sample")


def _plot_q3(input_path: Path, output_path: Path) -> None:
    data = np.loadtxt(input_path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise SystemExit(f"Expected five columns in {input_path}")

    r_values = np.asarray(data[:, 0], dtype=np.float64)
    s3_x = np.asarray(data[:, 1], dtype=np.float64)
    s3_y = np.asarray(data[:, 2], dtype=np.float64)
    s3_z = np.asarray(data[:, 3], dtype=np.float64)
    s3_avg = np.asarray(data[:, 4], dtype=np.float64)
    positive_r_mask = r_values > 0.0
    if not np.any(positive_r_mask):
        raise SystemExit("Need at least one positive r value to build the compensated plot.")

    r_plot = r_values[positive_r_mask]
    s3_x_comp = -s3_x[positive_r_mask] / r_plot
    s3_y_comp = -s3_y[positive_r_mask] / r_plot
    s3_z_comp = -s3_z[positive_r_mask] / r_plot
    s3_avg_comp = -s3_avg[positive_r_mask] / r_plot

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        ax.plot(r_plot, s3_x_comp, linewidth=1.5, label=r"$-S_{3,L}^{(x)}(r) / r$")
        ax.plot(r_plot, s3_y_comp, linewidth=1.5, label=r"$-S_{3,L}^{(y)}(r) / r$")
        ax.plot(r_plot, s3_z_comp, linewidth=1.5, label=r"$-S_{3,L}^{(z)}(r) / r$")
        ax.plot(r_plot, s3_avg_comp, linewidth=2.5, color="black", label=r"$-S_{3,L}^{\mathrm{avg}}(r) / r$")
        r_ref, plateau = _reference_plateau_q3(r_plot, s3_avg_comp)
        if r_ref is not None and plateau is not None:
            ax.plot(r_ref, plateau, linestyle="--", linewidth=1.5, color="0.3", label="Reference plateau")
        ax.axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$-S_{3,L}(r) / r$")
        ax.set_title("Compensated Third-Order Longitudinal Structure Function")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved structure-function plot: {output_path}")
    print("Order: q3")
    print("Plot scale: linear-linear")
    print("Plotted quantity: -S_3(r) / r")
    print("Reference guide: horizontal line anchored to the mid-range compensated average")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a saved second- or third-order isotropic longitudinal structure function file.",
    )
    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument("-q2", action="store_true", help="Plot the second-order structure function file")
    order_group.add_argument("-q3", action="store_true", help="Plot the third-order structure function file")
    parser.add_argument(
        "structure_function_path",
        help="Path to a *_spectra_structure_function.txt or *_structure_function_third_order.txt file",
    )
    parser.add_argument("--output", default=None, help="Optional output PDF path")
    args = parser.parse_args()

    input_path = Path(args.structure_function_path).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Structure-function file not found: {input_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)
    order = "q2" if args.q2 else "q3" if args.q3 else _infer_order_from_path(input_path)

    if order == "q2":
        _plot_q2(input_path, output_path)
    else:
        _plot_q3(input_path, output_path)


if __name__ == "__main__":
    main()
