from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI

from postprocess_fft.app import analyze_file_parallel


def read_structure_function_sections(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
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

    main = np.asarray(main_rows, dtype=np.float64)
    shell = np.asarray(shell_rows, dtype=np.float64) if shell_rows else None
    if main.ndim == 1:
        main = main.reshape(1, -1)
    if shell is not None and shell.ndim == 1:
        shell = shell.reshape(1, -1)
    return main, shell


def collect_metrics(input_path: Path) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix="third_order_rank_consistency_worker_") as tmpdir:
        work_path = Path(tmpdir) / input_path.name
        shutil.copy2(input_path, work_path)

        if MPI.COMM_WORLD.rank == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                analyze_file_parallel(
                    str(work_path),
                    MPI.COMM_WORLD,
                    backend_name="heffte_fftw",
                    visualize=False,
                )
        else:
            analyze_file_parallel(
                str(work_path),
                MPI.COMM_WORLD,
                backend_name="heffte_fftw",
                visualize=False,
            )
        MPI.COMM_WORLD.Barrier()

        if MPI.COMM_WORLD.rank != 0:
            return None

        structure_function_path = work_path.with_name(f"{work_path.stem}_spectra_structure_function.txt")
        if not structure_function_path.exists():
            raise FileNotFoundError(structure_function_path)

        table, shell_table = read_structure_function_sections(structure_function_path)
        if table.shape[1] < 6:
            raise ValueError(f"Expected at least six columns in {structure_function_path}, got {table.shape[1]}")
        if shell_table is None or shell_table.shape[1] < 3:
            raise ValueError(f"Expected a [shell] section with at least three columns in {structure_function_path}")

        return {
            "path": str(structure_function_path),
            "num_columns": int(table.shape[1]),
            "shell_num_columns": int(shell_table.shape[1]),
            "r": np.asarray(table[:, 0], dtype=np.float64).tolist(),
            "s2_l": np.asarray(table[:, 1], dtype=np.float64).tolist(),
            "s3_x": np.asarray(table[:, 2], dtype=np.float64).tolist(),
            "s3_y": np.asarray(table[:, 3], dtype=np.float64).tolist(),
            "s3_z": np.asarray(table[:, 4], dtype=np.float64).tolist(),
            "s3_avg": np.asarray(table[:, 5], dtype=np.float64).tolist(),
            "shell_r": np.asarray(shell_table[:, 0], dtype=np.float64).tolist(),
            "s3_shell": np.asarray(shell_table[:, 1], dtype=np.float64).tolist(),
            "shell_count": np.asarray(shell_table[:, 2], dtype=np.float64).tolist(),
        }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MPI worker for third-order rank-consistency tests.")
    parser.add_argument("input_path", help="Input TXT or HDF5 file")
    parser.add_argument("output_json", help="Rank-0 JSON output path")
    args = parser.parse_args()

    metrics = collect_metrics(Path(args.input_path).resolve())
    if MPI.COMM_WORLD.rank == 0:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
    MPI.COMM_WORLD.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
