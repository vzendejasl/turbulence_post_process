#!/usr/bin/env python3
"""Compare this repo's structure-function output against fastSF HDF5 outputs.

Examples:
  python tools/compare_structure_functions_code.py \
      tuo_data/post_process/cycle_179071/hit_out_Re800NumPtsPerDir256_fields_s9_write00417_spectra_structure_function.txt \
      tuo_data/structure_function_data_417 \
      -q 2

  python tools/compare_structure_functions_code.py \
      tuo_data/post_process/cycle_179071/hit_out_Re800NumPtsPerDir256_fields_s9_write00417_spectra_structure_function.txt \
      tuo_data/structure_function_data_417 \
      -q 3 --output compare_q3_417.pdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


def _plot_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }


def _read_structure_function_sections(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    main_rows: list[list[float]] = []
    shell_rows: list[list[float]] = []
    section = "main"

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "[main]":
                section = "main"
                continue
            if stripped == "[shell]":
                section = "shell"
                continue
            row = [float(value.strip()) for value in stripped.split(",")]
            if section == "shell":
                shell_rows.append(row)
            else:
                main_rows.append(row)

    if not main_rows:
        raise SystemExit(f"No [main] structure-function rows found in {path}")

    main = np.asarray(main_rows, dtype=np.float64)
    shell = np.asarray(shell_rows, dtype=np.float64) if shell_rows else None
    if main.ndim == 1:
        main = main.reshape(1, -1)
    if shell is not None and shell.ndim == 1:
        shell = shell.reshape(1, -1)
    return main, shell


def _resolve_fastsf_h5(path_hint: Path) -> Path:
    if path_hint.is_file():
        return path_hint
    candidate = path_hint / "SF_Grid_pll.h5"
    if candidate.exists():
        return candidate
    raise SystemExit(f"Could not find fastSF HDF5 file under {path_hint}")


def _fastsf_radial_average_3d(h5_path: Path, q_order: int) -> tuple[np.ndarray, np.ndarray]:
    dataset_name = f"SF_Grid_pll{int(q_order)}"
    with h5py.File(h5_path, "r") as handle:
        if dataset_name not in handle:
            raise SystemExit(f"Dataset {dataset_name} not found in {h5_path}")
        sf = np.asarray(handle[dataset_name][:], dtype=np.float64)

    nx, ny, nz = sf.shape
    nr = int(np.ceil(np.sqrt((nx - 1) ** 2 + (ny - 1) ** 2 + (nz - 1) ** 2))) + 1
    r_values = np.zeros(nr, dtype=np.float64)
    for i in range(nr):
        r_values[i] = np.sqrt(3.0) * i / (2.0 * nr)

    shell_sum = np.zeros(nr, dtype=np.float64)
    shell_count = np.zeros(nr, dtype=np.float64)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                shell_index = int(np.ceil(np.sqrt(x * x + y * y + z * z)))
                shell_sum[shell_index] += sf[x, y, z]
                shell_count[shell_index] += 1.0

    valid = shell_count > 0.0
    shell_sum[valid] /= shell_count[valid]
    return r_values, shell_sum


def _compensate_curve(r_values: np.ndarray, values: np.ndarray, q_order: int) -> tuple[np.ndarray, np.ndarray]:
    positive_r = np.asarray(r_values, dtype=np.float64) > 0.0
    r_plot = np.asarray(r_values[positive_r], dtype=np.float64)
    curve = np.asarray(values[positive_r], dtype=np.float64)

    if q_order == 2:
        return r_plot, curve * np.power(r_plot, -2.0 / 3.0)
    if q_order == 3:
        return r_plot, -curve * np.power(r_plot, -1.0)
    raise ValueError("q_order must be 2 or 3.")


def _interp_compare(reference_r: np.ndarray, reference_values: np.ndarray, other_r: np.ndarray, other_values: np.ndarray) -> tuple[float, float]:
    overlap_min = max(float(np.min(reference_r)), float(np.min(other_r)))
    overlap_max = min(float(np.max(reference_r)), float(np.max(other_r)))
    mask = (reference_r >= overlap_min) & (reference_r <= overlap_max)
    sample_r = reference_r[mask]
    if len(sample_r) == 0:
        return float("nan"), float("nan")

    interpolated = np.interp(sample_r, other_r, other_values)
    diff = interpolated - reference_values[mask]
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(reference_values[mask]), 1.0e-30))
    max_abs = float(np.max(np.abs(diff)))
    return rel_l2, max_abs


def _default_output_path(our_path: Path, q_order: int) -> Path:
    return our_path.with_name(f"{our_path.stem}_vs_fastsf_q{int(q_order)}.pdf")


def _default_linear_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_linear{output_path.suffix}")


def _reference_constant_line(r_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    valid = (np.asarray(r_values, dtype=np.float64) > 0.0) & np.isfinite(values) & (np.asarray(values, dtype=np.float64) > 0.0)
    if not np.any(valid):
        return None, None

    r_valid = np.asarray(r_values[valid], dtype=np.float64)
    values_valid = np.asarray(values[valid], dtype=np.float64)
    start = len(values_valid) // 3
    stop = max(start + 1, 2 * len(values_valid) // 3)
    level = float(np.median(values_valid[start:stop]))
    return r_valid, np.full_like(r_valid, level, dtype=np.float64)


def _reference_constant_line_linear(r_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    valid = (np.asarray(r_values, dtype=np.float64) > 0.0) & np.isfinite(values)
    if not np.any(valid):
        return None, None

    r_valid = np.asarray(r_values[valid], dtype=np.float64)
    values_valid = np.asarray(values[valid], dtype=np.float64)
    start = len(values_valid) // 3
    stop = max(start + 1, 2 * len(values_valid) // 3)
    level = float(np.median(values_valid[start:stop]))
    return r_valid, np.full_like(r_valid, level, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare this repo's structure functions against fastSF outputs.")
    parser.add_argument("our_structure_file", help="Path to *_spectra_structure_function.txt from this repo")
    parser.add_argument("fastsf_path", help="Path to fastSF output dir or SF_Grid_pll.h5")
    parser.add_argument("-q", "--order", type=int, choices=(2, 3), required=True, help="Structure-function order to compare")
    parser.add_argument("--output", default=None, help="Optional output PDF path")
    args = parser.parse_args()

    our_path = Path(args.our_structure_file).expanduser().resolve()
    fastsf_h5 = _resolve_fastsf_h5(Path(args.fastsf_path).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(our_path, args.order)
    linear_output_path = _default_linear_output_path(output_path)

    our_main, our_shell = _read_structure_function_sections(our_path)
    fastsf_r, fastsf_values = _fastsf_radial_average_3d(fastsf_h5, args.order)

    our_r = np.asarray(our_main[:, 0], dtype=np.float64)
    if args.order == 2:
        our_values = np.asarray(our_main[:, 1], dtype=np.float64)
        our_label = r"Our $S_{2,L}$"
    else:
        our_values = np.asarray(our_main[:, 5], dtype=np.float64)
        our_label = r"Our $S_{3,L}^{\mathrm{avg}}$"

    our_r_comp, our_values_comp = _compensate_curve(our_r, our_values, args.order)
    fastsf_r_comp, fastsf_values_comp = _compensate_curve(fastsf_r, fastsf_values, args.order)

    shell_r_comp = shell_values_comp = None
    if args.order == 3 and our_shell is not None and our_shell.shape[1] >= 3:
        shell_r = np.asarray(our_shell[:, 0], dtype=np.float64)
        shell_values = np.asarray(our_shell[:, 1], dtype=np.float64)
        shell_r_comp, shell_values_comp = _compensate_curve(shell_r, shell_values, args.order)

    overlap_mask = fastsf_r_comp <= 0.5 + 1.0e-14
    fastsf_r_overlap = fastsf_r_comp[overlap_mask]
    fastsf_values_overlap = fastsf_values_comp[overlap_mask]

    rel_l2_avg, max_abs_avg = _interp_compare(fastsf_r_overlap, fastsf_values_overlap, our_r_comp, our_values_comp)
    rel_l2_shell = max_abs_shell = float("nan")
    if shell_r_comp is not None and shell_values_comp is not None:
        rel_l2_shell, max_abs_shell = _interp_compare(fastsf_r_overlap, fastsf_values_overlap, shell_r_comp, shell_values_comp)

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        ax.loglog(fastsf_r_overlap, np.abs(fastsf_values_overlap), linewidth=2.5, color="tab:red", label="fastSF radial reduction")
        ax.loglog(our_r_comp, np.abs(our_values_comp), linewidth=2.0, color="black", label=our_label)
        if shell_r_comp is not None and shell_values_comp is not None:
            ax.loglog(shell_r_comp, np.abs(shell_values_comp), linewidth=1.8, linestyle="-.", color="tab:blue", label=r"Our $S_{3,L}^{\mathrm{shell}}$")
        ref_r, ref_values = _reference_constant_line(our_r_comp, np.abs(our_values_comp))
        if ref_r is not None and ref_values is not None:
            ax.loglog(ref_r, ref_values, linestyle="--", linewidth=1.3, color="0.25")

        ax.set_xlabel(r"$r$")
        if args.order == 2:
            ax.set_ylabel(r"$\left| r^{-2/3} S_2(r) \right|$")
            ax.set_title("Structure-Function Comparison: q=2")
        else:
            ax.set_ylabel(r"$\left| -S_3(r) / r \right|$")
            ax.set_title("Structure-Function Comparison: q=3")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        ax.plot(fastsf_r_overlap, fastsf_values_overlap, linewidth=2.5, color="tab:red", label="fastSF radial reduction")
        ax.plot(our_r_comp, our_values_comp, linewidth=2.0, color="black", label=our_label)
        if shell_r_comp is not None and shell_values_comp is not None:
            ax.plot(shell_r_comp, shell_values_comp, linewidth=1.8, linestyle="-.", color="tab:blue", label=r"Our $S_{3,L}^{\mathrm{shell}}$")
        ref_r, ref_values = _reference_constant_line_linear(our_r_comp, our_values_comp)
        if ref_r is not None and ref_values is not None:
            ax.plot(ref_r, ref_values, linestyle="--", linewidth=1.3, color="0.25")

        ax.set_xlabel(r"$r$")
        if args.order == 2:
            ax.set_ylabel(r"$r^{-2/3} S_2(r)$")
            ax.set_title("Structure-Function Comparison: q=2 (Linear)")
        else:
            ax.set_ylabel(r"$-S_3(r) / r$")
            ax.set_title("Structure-Function Comparison: q=3 (Linear)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(linear_output_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved comparison plot: {output_path}")
    print(f"Saved linear comparison plot: {linear_output_path}")
    print(f"fastSF HDF5: {fastsf_h5}")
    print(f"Our structure-function file: {our_path}")
    print("Comparison note: fastSF radial reduction is compared only over the overlap range r <= 0.5.")
    print(f"Relative L2 error vs fastSF using our main curve: {rel_l2_avg:.8e}")
    print(f"Max absolute difference vs fastSF using our main curve: {max_abs_avg:.8e}")
    if args.order == 3 and shell_r_comp is not None and shell_values_comp is not None:
        print(f"Relative L2 error vs fastSF using our shell curve: {rel_l2_shell:.8e}")
        print(f"Max absolute difference vs fastSF using our shell curve: {max_abs_shell:.8e}")


if __name__ == "__main__":
    main()
