#!/usr/bin/env python3
"""Plot a saved longitudinal structure function file to PDF.

Examples:
  python tools/plot_structure_function.py path/to/foo_spectra_structure_function.txt
  python tools/plot_structure_function.py path/to/foo_spectra_structure_function.txt --output foo_structure_function.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


def _default_output_path(input_path: Path) -> Path:
    suffix = "_structure_function"
    if input_path.stem.endswith(suffix):
        return input_path.with_name(f"{input_path.stem}_plot.pdf")
    return input_path.with_suffix(".pdf")


def _plot_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }


def _reference_plateau(r_values: np.ndarray, compensated_values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a saved isotropic longitudinal structure function file.",
    )
    parser.add_argument("structure_function_path", help="Path to a *_spectra_structure_function.txt file")
    parser.add_argument("--output", default=None, help="Optional output PDF path")
    args = parser.parse_args()

    input_path = Path(args.structure_function_path).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Structure-function file not found: {input_path}")

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

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        ax.loglog(
            r_values[positive_mask],
            compensated_values[positive_mask],
            linewidth=2.0,
            label=r"$r^{-2/3} S_L(r)$",
        )
        r_ref, plateau = _reference_plateau(r_values, compensated_values)
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
    print("Plot scale: log-log")
    print("Plotted quantity: r^(-2/3) * S_L(r)")
    print("Reference guide: horizontal line anchored to the mid-range compensated sample")


if __name__ == "__main__":
    main()
