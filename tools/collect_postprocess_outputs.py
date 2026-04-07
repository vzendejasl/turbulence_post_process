#!/usr/bin/env python3
"""Collect per-cycle post-processing outputs into a single post_process tree.

Examples:
  Copy outputs from one sampled-data root:
    python tools/collect_postprocess_outputs.py /path/to/blast_tgv3Dk2r5Re1600_SampledData

  Copy outputs from multiple sampled-data roots:
    python tools/collect_postprocess_outputs.py /path/to/run1_SampledData /path/to/run2_SampledData

  Move outputs instead of copying them:
    python tools/collect_postprocess_outputs.py /path/to/blast_tgv3Dk2r5Re1600_SampledData --move
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def cycle_directories(root: Path) -> list[Path]:
    """Return sorted cycle_* directories directly under one sampled-data root."""
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("cycle_")],
        key=lambda path: path.name,
    )


def spectra_files(cycle_dir: Path) -> list[Path]:
    """Return spectra text files stored directly in one cycle directory."""
    return sorted(cycle_dir.glob("*_spectra*.txt"))


def copy_or_move_path(source: Path, destination: Path, move: bool) -> None:
    """Copy or move one file or directory into place."""
    if not source.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        if move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copytree(source, destination)
        return

    if move:
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))
    else:
        shutil.copy2(source, destination)


def process_root(root: Path, move: bool) -> int:
    """Collect one sampled-data root into root/post_process."""
    if not root.exists():
        print(f"Skipping missing root: {root}")
        return 1
    if not root.is_dir():
        print(f"Skipping non-directory root: {root}")
        return 1

    cycles = cycle_directories(root)
    if not cycles:
        print(f"No cycle_* directories found under: {root}")
        return 0

    postprocess_root = root / "post_process"
    postprocess_root.mkdir(parents=True, exist_ok=True)

    print(f"Processing root: {root}")
    print(f"  Found {len(cycles)} cycle directories")
    print(f"  Destination: {postprocess_root}")

    copied_items = 0

    for cycle_dir in cycles:
        destination_cycle = postprocess_root / cycle_dir.name
        destination_cycle.mkdir(parents=True, exist_ok=True)

        slice_data_dir = cycle_dir / "slice_data"
        if slice_data_dir.exists():
            destination_slice_data = destination_cycle / "slice_data"
            copy_or_move_path(slice_data_dir, destination_slice_data, move)
            copied_items += 1
            print(f"  {'Moved' if move else 'Copied'} {slice_data_dir} -> {destination_slice_data}")

        slice_plots_dir = cycle_dir / "slice_plots"
        if slice_plots_dir.exists():
            destination_slice_plots = destination_cycle / "slice_plots"
            copy_or_move_path(slice_plots_dir, destination_slice_plots, move)
            copied_items += 1
            print(f"  {'Moved' if move else 'Copied'} {slice_plots_dir} -> {destination_slice_plots}")

        for spectra_path in spectra_files(cycle_dir):
            destination_spectra = destination_cycle / spectra_path.name
            copy_or_move_path(spectra_path, destination_spectra, move)
            copied_items += 1
            print(f"  {'Moved' if move else 'Copied'} {spectra_path} -> {destination_spectra}")

    print(f"  Completed {root} with {copied_items} copied item(s)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect slice_data, slice_plots, and spectra outputs from cycle_* directories into a post_process tree."
    )
    parser.add_argument("roots", nargs="+", help="One or more sampled-data root directories")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move outputs instead of copying them. Default behavior is copy-only.",
    )
    args = parser.parse_args()

    failures = 0
    for root_arg in args.roots:
        failures += process_root(Path(root_arg).resolve(), move=args.move)
        print()

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
