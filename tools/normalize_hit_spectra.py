#!/usr/bin/env python3
"""Normalize saved forced-HIT total spectra with AvgDissipation and AvgKolmLen.

Examples:
  python tools/normalize_hit_spectra.py /path/to/post_process/cycle_58905
  python tools/normalize_hit_spectra.py /path/to/post_process
  python tools/normalize_hit_spectra.py /path/to/post_process/cycle_* --no-plot
  python tools/normalize_hit_spectra.py /path/to/post_process/cycle_58905 --k-cutoff 128

The script expects each cycle directory to contain one or more saved
``*_spectra.txt`` files plus matching ``*_spectra_metadata.txt`` files. It
finds the nearest ``hit_out_turb_*.csv`` up the directory tree, matches the
cycle/step number, and uses:

  - ``AvgDissipation`` for epsilon
  - ``AvgKolmLen`` for eta

It writes one normalized text file and, unless ``--no-plot`` is given, one PDF
plot beside each input spectrum. The script currently normalizes only the saved
total spectrum ``E_total``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_TURB_CSV_PATTERNS = (
    "hit_out_turb_*.csv",
    "tgv_out_turb_*.csv",
)


_CSV_COLUMN_ALIASES = {
    "Cycle": {
        "cycle",
        "step",
    },
    "AvgDissipation": {
        "avgdissipation",
        "averagedissipation",
    },
    "AvgKolmLen": {
        "avgkolmlen",
        "averagekolmlen",
        "avgkolmogorovlengthscale",
        "averagekolmogorovlengthscale",
    },
}


def _plot_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _collect_cycle_directories(inputs: list[str]) -> list[Path]:
    cycle_dirs: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise SystemExit(f"Input path does not exist: {path}")
        if not path.is_dir():
            raise SystemExit(f"Input path is not a directory: {path}")

        if path.name.startswith("cycle_"):
            cycle_dirs.append(path)
            continue

        direct_cycles = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_dir() and candidate.name.startswith("cycle_")
        )
        if direct_cycles:
            cycle_dirs.extend(direct_cycles)
            continue

        post_process = path / "post_process"
        if post_process.is_dir():
            nested_cycles = sorted(
                candidate
                for candidate in post_process.iterdir()
                if candidate.is_dir() and candidate.name.startswith("cycle_")
            )
            if nested_cycles:
                cycle_dirs.extend(nested_cycles)
                continue

        raise SystemExit(
            f"No cycle_* directories found under: {path}"
        )

    return sorted(_unique_paths(cycle_dirs))


def _cycle_number_from_name(cycle_dir: Path) -> int:
    match = re.fullmatch(r"cycle_(\d+)", cycle_dir.name)
    if match is None:
        raise SystemExit(f"Could not parse cycle number from directory name: {cycle_dir}")
    return int(match.group(1))


def _spectra_files(cycle_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in cycle_dir.glob("*_spectra.txt")
        if path.is_file() and not path.name.endswith("_metadata.txt")
    )


def _metadata_path_for_spectrum(spectrum_path: Path) -> Path:
    metadata_path = spectrum_path.with_name(f"{spectrum_path.stem}_metadata.txt")
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata file for spectrum: {metadata_path}")
    return metadata_path


def _read_step_number(metadata_path: Path) -> int | None:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    match = re.search(r"(?:Step|Cycle)\s*[:=]\s*(\d+)", first_line)
    if match is None:
        return None
    return int(match.group(1))


def _candidate_csv_match(csv_path: Path, spectrum_path: Path) -> bool:
    spectrum_name = spectrum_path.name
    tokens = re.findall(r"Re\d+|NumPtsPerDir\d+", spectrum_name)
    if not tokens:
        return True
    return all(token in csv_path.name for token in tokens)


def _normalize_csv_header_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _resolve_required_csv_columns(header: list[str]) -> dict[str, int] | None:
    normalized_to_index = {
        _normalize_csv_header_name(name): index
        for index, name in enumerate(header)
    }
    resolved: dict[str, int] = {}
    for canonical_name, aliases in _CSV_COLUMN_ALIASES.items():
        for alias in aliases:
            index = normalized_to_index.get(alias)
            if index is not None:
                resolved[canonical_name] = index
                break
        else:
            return None
    return resolved


def _csv_has_required_turbulence_columns(csv_path: Path) -> bool:
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            raw_header = next(reader)
    except (OSError, StopIteration):
        return False

    header = [entry.strip() for entry in raw_header]
    return _resolve_required_csv_columns(header) is not None


def _find_turbulence_csv(cycle_dir: Path, spectrum_path: Path, explicit_csv: Path | None) -> Path:
    if explicit_csv is not None:
        if not explicit_csv.is_file():
            raise SystemExit(f"Specified turbulence CSV does not exist: {explicit_csv}")
        return explicit_csv.resolve()

    for ancestor in [cycle_dir, *cycle_dir.parents]:
        matches = sorted(
            {
                path.resolve()
                for pattern in _TURB_CSV_PATTERNS
                for path in ancestor.glob(pattern)
                if path.is_file()
            }
        )
        if not matches:
            continue
        filtered = [path for path in matches if _candidate_csv_match(path, spectrum_path)]
        filtered_with_columns = [
            path for path in filtered if _csv_has_required_turbulence_columns(path)
        ]
        matches_with_columns = [
            path for path in matches if _csv_has_required_turbulence_columns(path)
        ]
        if len(filtered_with_columns) == 1:
            return filtered_with_columns[0].resolve()
        if len(matches_with_columns) == 1:
            return matches_with_columns[0].resolve()
        if len(filtered) == 1:
            return filtered[0].resolve()
        if len(matches) == 1:
            return matches[0].resolve()
        if filtered_with_columns:
            raise SystemExit(
                "Found multiple candidate turbulence CSV files with the required columns near "
                f"{cycle_dir}; use --turb-csv to disambiguate: "
                + ", ".join(str(path) for path in filtered_with_columns)
            )
        if matches_with_columns:
            raise SystemExit(
                "Found multiple turbulence CSV files with the required columns near "
                f"{cycle_dir}; use --turb-csv to choose one: "
                + ", ".join(str(path) for path in matches_with_columns)
            )
        if filtered:
            raise SystemExit(
                "Found multiple candidate turbulence CSV files near "
                f"{cycle_dir}; use --turb-csv to disambiguate: "
                + ", ".join(str(path) for path in filtered)
            )
        raise SystemExit(
            "Found multiple turbulence CSV files near "
            f"{cycle_dir}; use --turb-csv to choose one: "
            + ", ".join(str(path) for path in matches)
        )

    raise SystemExit(
        "Could not find a supported turbulence CSV file above cycle directory: "
        f"{cycle_dir}"
    )


def _read_csv_row_by_cycle(csv_path: Path, cycle_number: int) -> dict[str, float]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise SystemExit(f"CSV file is empty: {csv_path}") from exc

        header = [entry.strip() for entry in raw_header]
        indices = _resolve_required_csv_columns(header)
        if indices is None:
            raise SystemExit(
                "Missing required turbulence columns in CSV file: "
                f"{csv_path}. Need columns compatible with Cycle/cycle, "
                "AvgDissipation/Average Dissipation, and AvgKolmLen/Average Kolm Len."
            )

        for raw_row in reader:
            if not raw_row:
                continue
            row = [entry.strip() for entry in raw_row]
            try:
                row_cycle = int(round(float(row[indices["Cycle"]])))
            except (ValueError, IndexError):
                continue
            if row_cycle != cycle_number:
                continue
            try:
                return {
                    "Cycle": float(row[indices["Cycle"]]),
                    "AvgDissipation": float(row[indices["AvgDissipation"]]),
                    "AvgKolmLen": float(row[indices["AvgKolmLen"]]),
                }
            except (ValueError, IndexError) as exc:
                raise SystemExit(
                    f"Could not parse AvgDissipation/AvgKolmLen for cycle {cycle_number} in {csv_path}"
                ) from exc

    raise SystemExit(f"Could not find cycle {cycle_number} in CSV file: {csv_path}")


def _read_spectrum(spectrum_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(spectrum_path, delimiter=",", skiprows=1)
    except ValueError as exc:
        raise SystemExit(f"Could not read spectrum file: {spectrum_path}") from exc

    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise SystemExit(f"Expected at least two columns in spectrum file: {spectrum_path}")
    return data[:, 0], data[:, 1]


def _normalized_output_paths(spectrum_path: Path) -> tuple[Path, Path, Path]:
    stem = spectrum_path.stem
    text_path = spectrum_path.with_name(f"{stem}_normalized.txt")
    metadata_path = spectrum_path.with_name(f"{stem}_normalized_metadata.txt")
    pdf_path = spectrum_path.with_name(f"{stem}_normalized.pdf")
    return text_path, metadata_path, pdf_path


def _compute_normalized_spectrum(
    k_saved: np.ndarray,
    e_total: np.ndarray,
    epsilon: float,
    eta: float,
    domain_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if epsilon <= 0.0:
        raise SystemExit(f"AvgDissipation must be positive, got {epsilon:.16e}")
    if eta <= 0.0:
        raise SystemExit(f"AvgKolmLen must be positive, got {eta:.16e}")
    if domain_length <= 0.0:
        raise SystemExit(f"Domain length must be positive, got {domain_length:.16e}")

    delta_k_phys = 2.0 * math.pi / domain_length
    k_phys = delta_k_phys * np.asarray(k_saved, dtype=np.float64)
    k_eta = k_phys * eta
    e_density = np.asarray(e_total, dtype=np.float64) / delta_k_phys
    energy_scale = (epsilon ** (2.0 / 3.0)) * (eta ** (5.0 / 3.0))
    e_normalized = e_density / energy_scale
    e_compensated = np.zeros_like(e_normalized, dtype=np.float64)
    positive_mask = k_eta > 0.0
    e_compensated[positive_mask] = (
        e_normalized[positive_mask] * (k_eta[positive_mask] ** (5.0 / 3.0))
    )
    return k_eta, e_normalized, e_compensated, float(delta_k_phys)


def _apply_k_cutoff(
    k_saved: np.ndarray,
    e_total: np.ndarray,
    k_cutoff: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if k_cutoff is None:
        return np.asarray(k_saved, dtype=np.float64), np.asarray(e_total, dtype=np.float64)
    if k_cutoff < 0:
        raise SystemExit(f"k cutoff must be non-negative, got {k_cutoff}")

    k_saved = np.asarray(k_saved, dtype=np.float64)
    e_total = np.asarray(e_total, dtype=np.float64)
    mask = k_saved <= float(k_cutoff)
    if not np.any(mask):
        raise SystemExit(f"No spectrum entries remain after applying k cutoff <= {k_cutoff}")
    return k_saved[mask], e_total[mask]


def _write_normalized_text(
    output_path: Path,
    k_eta: np.ndarray,
    e_normalized: np.ndarray,
    e_compensated: np.ndarray,
) -> None:
    header_labels = [
        "k_eta",
        "E_density_norm",
        "E_density_comp_norm",
    ]
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(", ".join(f"{label:>23s}" for label in header_labels) + "\n")
        for row in zip(k_eta, e_normalized, e_compensated):
            handle.write(", ".join(f"{value:>23.16e}" for value in row) + "\n")


def _write_normalized_metadata(
    output_path: Path,
    spectrum_path: Path,
    csv_path: Path,
    cycle_number: int,
    epsilon: float,
    eta: float,
    domain_length: float,
    delta_k_phys: float,
    k_cutoff: int | None,
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        kolmogorov_energy_scale = (epsilon ** (2.0 / 3.0)) * (eta ** (5.0 / 3.0))
        handle.write(f"# Source spectrum: {spectrum_path}\n")
        handle.write(f"# Turbulence CSV: {csv_path}\n")
        handle.write(f"# Cycle: {cycle_number}\n")
        handle.write(f"# AvgDissipation: {epsilon:.16e}\n")
        handle.write(f"# AvgKolmLen: {eta:.16e}\n")
        handle.write(f"# Domain length: {domain_length:.16e}\n")
        handle.write(f"# Physical shell width delta_k: {delta_k_phys:.16e}\n")
        if k_cutoff is None:
            handle.write("# Integer-wavenumber cutoff: none\n")
        else:
            handle.write(f"# Integer-wavenumber cutoff: k_saved <= {k_cutoff}\n")
        handle.write(
            f"# Spectrum normalization scale epsilon^(2/3) * eta^(5/3): "
            f"{kolmogorov_energy_scale:.16e}\n"
        )
        handle.write(
            "# Physical spectrum density from saved shell energy: "
            "E_density = E_total / delta_k\n"
        )
        handle.write(
            "# Uncompensated normalized spectrum: "
            "E_density_norm = E_density / (epsilon^(2/3) * eta^(5/3))\n"
        )
        handle.write(
            "# Compensated normalized spectrum: "
            "E_density_comp_norm = E_density_norm * (k_eta)^(5/3)\n"
        )
        handle.write(
            "# Data columns: k_eta, E_density_norm, E_density_comp_norm\n"
        )


def _positive_curve(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (np.asarray(x_values, dtype=np.float64) > 0.0)
        & (np.asarray(y_values, dtype=np.float64) > 0.0)
    )
    return np.asarray(x_values[mask], dtype=np.float64), np.asarray(y_values[mask], dtype=np.float64)


def _plot_normalized_spectrum(
    output_path: Path,
    cycle_number: int,
    epsilon: float,
    eta: float,
    k_eta: np.ndarray,
    e_normalized: np.ndarray,
    e_compensated: np.ndarray,
) -> None:
    x_uncomp, y_uncomp = _positive_curve(k_eta, e_normalized)
    x_comp, y_comp = _positive_curve(k_eta, e_compensated)
    if len(x_uncomp) == 0:
        raise SystemExit("Need at least one positive (k*eta, E_normalized) pair for plotting.")
    if len(x_comp) == 0:
        raise SystemExit("Need at least one positive (k*eta, compensated spectrum) pair for plotting.")

    with plt.rc_context(_plot_style()):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 8.0), sharex=True)

        ax0.loglog(x_uncomp, y_uncomp, linewidth=2.0, color="tab:blue")
        ax0.set_ylabel(r"$E(k) / (\epsilon^{2/3}\eta^{5/3})$")
        ax0.set_title(
            "Normalized HIT Total Spectrum\n"
            rf"cycle={cycle_number}, $\epsilon={epsilon:.3e}$, $\eta={eta:.3e}$"
        )
        ax0.grid(True, which="both", alpha=0.3)

        ax1.semilogx(x_comp, y_comp, linewidth=2.0, color="tab:red")
        ax1.set_xlabel(r"$k\eta$")
        ax1.set_ylabel(
            r"$E(k) / (\epsilon^{2/3}\eta^{5/3}) \, (k\eta)^{5/3}$"
        )
        ax1.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def _process_spectrum(
    spectrum_path: Path,
    cycle_dir: Path,
    domain_length: float,
    k_cutoff: int | None,
    turb_csv_override: Path | None,
    make_plot: bool,
) -> None:
    metadata_path = _metadata_path_for_spectrum(spectrum_path)
    cycle_number = _read_step_number(metadata_path)
    if cycle_number is None:
        cycle_number = _cycle_number_from_name(cycle_dir)

    csv_path = _find_turbulence_csv(cycle_dir, spectrum_path, turb_csv_override)
    stats = _read_csv_row_by_cycle(csv_path, cycle_number)
    k_saved, e_total = _read_spectrum(spectrum_path)
    k_saved, e_total = _apply_k_cutoff(k_saved, e_total, k_cutoff)
    k_eta, e_normalized, e_compensated, delta_k_phys = _compute_normalized_spectrum(
        k_saved=k_saved,
        e_total=e_total,
        epsilon=stats["AvgDissipation"],
        eta=stats["AvgKolmLen"],
        domain_length=domain_length,
    )

    text_path, metadata_path, pdf_path = _normalized_output_paths(spectrum_path)
    _write_normalized_text(
        output_path=text_path,
        k_eta=k_eta,
        e_normalized=e_normalized,
        e_compensated=e_compensated,
    )
    _write_normalized_metadata(
        output_path=metadata_path,
        spectrum_path=spectrum_path,
        csv_path=csv_path,
        cycle_number=cycle_number,
        epsilon=stats["AvgDissipation"],
        eta=stats["AvgKolmLen"],
        domain_length=domain_length,
        delta_k_phys=delta_k_phys,
        k_cutoff=k_cutoff,
    )

    if make_plot:
        _plot_normalized_spectrum(
            output_path=pdf_path,
            cycle_number=cycle_number,
            epsilon=stats["AvgDissipation"],
            eta=stats["AvgKolmLen"],
            k_eta=k_eta,
            e_normalized=e_normalized,
            e_compensated=e_compensated,
        )

    print(f"Cycle directory: {cycle_dir}")
    print(f"  Spectrum: {spectrum_path.name}")
    print(f"  Turbulence CSV: {csv_path}")
    print(f"  Cycle: {cycle_number}")
    print(f"  Normalization dissipation (AvgDissipation): {stats['AvgDissipation']:.16e}")
    print(f"  Normalization Kolmogorov length (AvgKolmLen): {stats['AvgKolmLen']:.16e}")
    print(f"  Domain length used for k conversion: {domain_length:.16e}")
    if k_cutoff is not None:
        print(f"  Integer-wavenumber cutoff: <= {k_cutoff}")
    print(f"  Wrote: {text_path}")
    print(f"  Wrote: {metadata_path}")
    if make_plot:
        print(f"  Wrote: {pdf_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize saved HIT total spectra in one or more cycle_* directories "
            "using dissipation and Kolmogorov-length columns from a supported "
            "turbulence CSV."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more cycle_* directories, a post_process directory, or a "
            "directory that contains post_process/."
        ),
    )
    parser.add_argument(
        "--domain-length",
        type=float,
        default=1.0,
        help=(
            "Physical box length L used to convert saved shell indices to "
            "physical k=(2*pi/L)*k_saved. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--turb-csv",
        type=Path,
        default=None,
        help="Optional override path to a supported turbulence CSV.",
    )
    parser.add_argument(
        "--k-cutoff",
        type=int,
        default=None,
        help=(
            "Optional cutoff on the saved integer shell index k_saved. "
            "Only rows with k_saved <= k_cutoff are kept."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write normalized text only; skip the PDF plot.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cycle_dirs = _collect_cycle_directories(args.paths)
    for cycle_dir in cycle_dirs:
        spectra = _spectra_files(cycle_dir)
        if not spectra:
            print(f"No *_spectra.txt files found under: {cycle_dir}")
            continue
        for spectrum_path in spectra:
            _process_spectrum(
                spectrum_path=spectrum_path,
                cycle_dir=cycle_dir,
                domain_length=float(args.domain_length),
                k_cutoff=args.k_cutoff,
                turb_csv_override=args.turb_csv,
                make_plot=not args.no_plot,
            )


if __name__ == "__main__":
    main()
