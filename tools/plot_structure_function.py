#!/usr/bin/env python3
"""Plot saved second- or third-order longitudinal structure function files to PDF.

Examples:
  python tools/plot_structure_function.py -q2 path/to/foo_spectra_structure_function.txt
  python tools/plot_structure_function.py -q3 path/to/foo_spectra_structure_function.txt
  python tools/plot_structure_function.py -q2 path/to/foo_spectra_structure_function.txt --plot-linear
  python tools/plot_structure_function.py -q3 path/to/foo_spectra_structure_function.txt --plot-linear --uncompensated
  python tools/plot_structure_function.py -q2 path/to/foo_spectra_structure_function.txt --output foo_structure_function.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


def _default_output_path(input_path: Path) -> Path:
    if input_path.stem.endswith("_structure_function_shell_third_order"):
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
    positive_mask = (r_values > 0.0) & np.isfinite(compensated_values) & (compensated_values > 0.0)
    if not np.any(positive_mask):
        return None, None

    r_positive = np.asarray(r_values[positive_mask], dtype=np.float64)
    comp_positive = np.asarray(compensated_values[positive_mask], dtype=np.float64)
    anchor_index = len(comp_positive) // 2
    anchor_value = float(comp_positive[anchor_index])
    return r_positive, np.full_like(r_positive, anchor_value, dtype=np.float64)


def _positive_loglog_curve(r_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (np.asarray(r_values, dtype=np.float64) > 0.0) & np.isfinite(values) & (np.asarray(values, dtype=np.float64) > 0.0)
    return np.asarray(r_values[mask], dtype=np.float64), np.asarray(values[mask], dtype=np.float64)


def _finite_linear_curve(r_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(r_values) & np.isfinite(values)
    return np.asarray(r_values[mask], dtype=np.float64), np.asarray(values[mask], dtype=np.float64)


def _plot_curve(ax, r_values: np.ndarray, values: np.ndarray, plot_scale: str, **kwargs) -> None:
    if plot_scale == "linear":
        ax.plot(r_values, values, **kwargs)
    else:
        ax.loglog(r_values, values, **kwargs)


def _scale_label(plot_scale: str) -> str:
    return "linear-linear" if plot_scale == "linear" else "log-log"


def _curve_builder(plot_scale: str):
    return _finite_linear_curve if plot_scale == "linear" else _positive_loglog_curve


def _infer_order_from_path(input_path: Path) -> str:
    return "q2"


def _read_structure_function_sections(input_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    main_rows: list[list[float]] = []
    shell_rows: list[list[float]] = []
    current_section = "main"

    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "[main]":
                current_section = "main"
                continue
            if stripped == "[shell]":
                current_section = "shell"
                continue

            row = [float(value.strip()) for value in stripped.split(",")]
            if current_section == "shell":
                shell_rows.append(row)
            else:
                main_rows.append(row)

    if not main_rows:
        raise SystemExit(f"No main structure-function rows found in {input_path}")

    main = np.asarray(main_rows, dtype=np.float64)
    shell = np.asarray(shell_rows, dtype=np.float64) if shell_rows else None
    if main.ndim == 1:
        main = main.reshape(1, -1)
    if shell is not None and shell.ndim == 1:
        shell = shell.reshape(1, -1)
    return main, shell


def _plot_q2(input_path: Path, output_path: Path, plot_scale: str, uncompensated: bool) -> None:
    data, _ = _read_structure_function_sections(input_path)
    if data.shape[1] < 2:
        raise SystemExit(f"Expected at least two columns in {input_path}")

    r_values = np.asarray(data[:, 0], dtype=np.float64)
    s_values = np.asarray(data[:, 1], dtype=np.float64)
    plot_values = np.asarray(s_values, dtype=np.float64)
    y_label = r"$S_L(r)$"
    title = "Longitudinal Second-Order Structure Function"
    reference_label = None
    reference_r = reference_values = None

    if not uncompensated:
        plot_values = np.zeros_like(s_values, dtype=np.float64)
        positive_r_mask = r_values > 0.0
        plot_values[positive_r_mask] = (
            s_values[positive_r_mask] * (r_values[positive_r_mask] ** (-2.0 / 3.0))
        )
        y_label = r"$r^{-2/3} S_L(r)$"
        title = "Compensated Longitudinal Second-Order Structure Function"
        reference_r, reference_values = _reference_plateau_q2(r_values, plot_values)
        reference_label = r"$\propto r^{2/3}$ reference"

    curve_builder = _curve_builder(plot_scale)
    if plot_scale == "linear":
        r_plot, values_plot = curve_builder(r_values, plot_values)
        if len(r_plot) == 0:
            raise SystemExit("Need at least one finite sample for a linear plot.")
    else:
        r_plot, values_plot = curve_builder(r_values, plot_values)
        if len(r_plot) == 0:
            if uncompensated:
                raise SystemExit("Need at least one positive (r, S_L) pair for an uncompensated log-log plot.")
            raise SystemExit("Need at least one positive (r, r^(-2/3) S_L) pair for a log-log plot.")

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        _plot_curve(
            ax,
            r_plot,
            values_plot,
            plot_scale,
            linewidth=2.0,
            label=y_label,
        )
        if reference_r is not None and reference_values is not None and reference_label is not None:
            _plot_curve(
                ax,
                reference_r,
                reference_values,
                plot_scale,
                linestyle="--",
                linewidth=1.5,
                label=reference_label,
            )

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if plot_scale == "loglog":
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved structure-function plot: {output_path}")
    print("Order: q2")
    print(f"Plot scale: {_scale_label(plot_scale)}")
    print(f"Plotted quantity: {'S_L(r)' if uncompensated else 'r^(-2/3) * S_L(r)'}")
    if not uncompensated:
        print("Reference guide: horizontal line anchored to the mid-range compensated sample")


def _plot_q3(input_path: Path, output_path: Path, with_shell: bool, plot_scale: str, uncompensated: bool) -> None:
    data, shell_section = _read_structure_function_sections(input_path)
    if data.shape[1] >= 6:
        r_values = np.asarray(data[:, 0], dtype=np.float64)
        s3_x = np.asarray(data[:, 2], dtype=np.float64)
        s3_y = np.asarray(data[:, 3], dtype=np.float64)
        s3_z = np.asarray(data[:, 4], dtype=np.float64)
        s3_avg = np.asarray(data[:, 5], dtype=np.float64)
        if with_shell and shell_section is not None and shell_section.shape[1] >= 3:
            s3_shell = np.asarray(shell_section[:, 1], dtype=np.float64)
            r_shell = np.asarray(shell_section[:, 0], dtype=np.float64)
            has_shell = True
        else:
            s3_shell = None
            r_shell = None
            has_shell = False
    elif data.shape[1] >= 5:
        r_values = np.asarray(data[:, 0], dtype=np.float64)
        s3_x = np.asarray(data[:, 1], dtype=np.float64)
        s3_y = np.asarray(data[:, 2], dtype=np.float64)
        s3_z = np.asarray(data[:, 3], dtype=np.float64)
        s3_avg = np.asarray(data[:, 4], dtype=np.float64)
        s3_shell = None
        has_shell = False
    else:
        raise SystemExit(f"Expected either 5 columns (legacy q3) or 6 columns (combined file) in {input_path}")

    positive_r_mask = r_values > 0.0
    if not np.any(positive_r_mask):
        raise SystemExit("Need at least one positive r value to build the plot.")

    if uncompensated and plot_scale != "linear":
        raise SystemExit("Uncompensated q3 plots use signed values, so please add --plot-linear.")

    r_plot = r_values[positive_r_mask]
    if uncompensated:
        s3_x_plot = s3_x[positive_r_mask]
        s3_y_plot = s3_y[positive_r_mask]
        s3_z_plot = s3_z[positive_r_mask]
        s3_avg_plot = s3_avg[positive_r_mask]
        y_label = r"$S_{3,L}(r)$"
        title = "Third-Order Longitudinal Structure Function"
        reference_r = reference_values = None
    else:
        s3_x_plot = np.abs(-s3_x[positive_r_mask] / r_plot)
        s3_y_plot = np.abs(-s3_y[positive_r_mask] / r_plot)
        s3_z_plot = np.abs(-s3_z[positive_r_mask] / r_plot)
        s3_avg_plot = np.abs(-s3_avg[positive_r_mask] / r_plot)
        y_label = r"$\left|-S_{3,L}(r) / r\right|$"
        title = "Compensated Third-Order Longitudinal Structure Function Magnitude"
        reference_r = r_plot if s3_avg_plot is not None else None
        reference_values = s3_avg_plot if s3_avg_plot is not None else None

    if has_shell and s3_shell is not None:
        if 'r_shell' not in locals() or r_shell is None:
            shell_mask = positive_r_mask & np.isfinite(s3_shell)
            r_shell = r_values[shell_mask]
            if uncompensated:
                s3_shell_plot = s3_shell[shell_mask]
            else:
                s3_shell_plot = np.abs(-s3_shell[shell_mask] / r_shell)
        else:
            shell_mask = (r_shell > 0.0) & np.isfinite(s3_shell)
            r_shell = r_shell[shell_mask]
            if uncompensated:
                s3_shell_plot = s3_shell[shell_mask]
            else:
                s3_shell_plot = np.abs(-s3_shell[shell_mask] / r_shell)
    else:
        r_shell = None
        s3_shell_plot = None

    curve_builder = _curve_builder(plot_scale)

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        if s3_avg_plot is not None:
            rx, vx = curve_builder(r_plot, s3_x_plot)
            ry, vy = curve_builder(r_plot, s3_y_plot)
            rz, vz = curve_builder(r_plot, s3_z_plot)
            ravg, vavg = curve_builder(r_plot, s3_avg_plot)
            if len(rx) > 0:
                _plot_curve(
                    ax,
                    rx,
                    vx,
                    plot_scale,
                    linewidth=1.5,
                    label=(r"$S_{3,L}^{(x)}(r)$" if uncompensated else r"$\left|-S_{3,L}^{(x)}(r) / r\right|$"),
                )
            if len(ry) > 0:
                _plot_curve(
                    ax,
                    ry,
                    vy,
                    plot_scale,
                    linewidth=1.5,
                    label=(r"$S_{3,L}^{(y)}(r)$" if uncompensated else r"$\left|-S_{3,L}^{(y)}(r) / r\right|$"),
                )
            if len(rz) > 0:
                _plot_curve(
                    ax,
                    rz,
                    vz,
                    plot_scale,
                    linewidth=1.5,
                    label=(r"$S_{3,L}^{(z)}(r)$" if uncompensated else r"$\left|-S_{3,L}^{(z)}(r) / r\right|$"),
                )
            if len(ravg) > 0:
                _plot_curve(
                    ax,
                    ravg,
                    vavg,
                    plot_scale,
                    linewidth=2.5,
                    color="black",
                    label=(r"$S_{3,L}^{\mathrm{avg}}(r)$" if uncompensated else r"$\left|-S_{3,L}^{\mathrm{avg}}(r) / r\right|$"),
                )
        if r_shell is not None and s3_shell_plot is not None and len(r_shell) > 0:
            rshell, vshell = curve_builder(r_shell, s3_shell_plot)
            if len(rshell) > 0:
                _plot_curve(
                    ax,
                    rshell,
                    vshell,
                    plot_scale,
                    linewidth=2.0,
                    linestyle="-.",
                    color="tab:red",
                    label=(r"$S_{3,L}^{\mathrm{shell}}(r)$" if uncompensated else r"$\left|-S_{3,L}^{\mathrm{shell}}(r) / r\right|$"),
                )
        reference_curve_values = reference_values if reference_values is not None else s3_shell_plot
        reference_curve_r = reference_r if reference_r is not None else r_shell
        r_ref, plateau = _reference_plateau_q3(reference_curve_r, reference_curve_values)
        if (not uncompensated) and r_ref is not None and plateau is not None:
            _plot_curve(
                ax,
                r_ref,
                plateau,
                plot_scale,
                linestyle="--",
                linewidth=1.5,
                color="0.3",
                label="Reference plateau",
            )
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if plot_scale == "loglog":
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved structure-function plot: {output_path}")
    print("Order: q3")
    print(f"Plot scale: {_scale_label(plot_scale)}")
    print(f"Plotted quantity: {'S_3(r)' if uncompensated else '| -S_3(r) / r |'}")
    if with_shell:
        print("Curves: axis-aligned average/direct curves, plus shell curve when a shell section is available")
    else:
        print("Curves: axis-aligned average/direct curves only")
    if not uncompensated:
        print("Reference guide: horizontal plateau anchored to the mid-range compensated magnitude")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a saved second- or third-order isotropic longitudinal structure function file.",
    )
    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument("-q2", action="store_true", help="Plot the second-order structure function file")
    order_group.add_argument("-q3", action="store_true", help="Plot the third-order structure function file")
    parser.add_argument(
        "structure_function_path",
        help="Path to a combined *_spectra_structure_function.txt file or a legacy third-order file",
    )
    parser.add_argument("--output", default=None, help="Optional output PDF path")
    parser.add_argument(
        "--plot-linear",
        action="store_true",
        help="Use linear-linear axes instead of the default log-log axes.",
    )
    parser.add_argument("--uncompensated", action="store_true", help="Plot the raw structure function instead of the compensated quantity.")
    parser.add_argument("--no-shell", action="store_true", help="Hide the shell-averaged q3 curve")
    args = parser.parse_args()

    input_path = Path(args.structure_function_path).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Structure-function file not found: {input_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)
    order = "q2" if args.q2 else "q3" if args.q3 else _infer_order_from_path(input_path)
    plot_scale = "linear" if args.plot_linear else "loglog"

    if order == "q2":
        _plot_q2(input_path, output_path, plot_scale, uncompensated=args.uncompensated)
    else:
        _plot_q3(
            input_path,
            output_path,
            with_shell=not args.no_shell,
            plot_scale=plot_scale,
            uncompensated=args.uncompensated,
        )


if __name__ == "__main__":
    main()
