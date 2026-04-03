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


def collect_metrics(input_path: Path) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix="rank_consistency_worker_") as tmpdir:
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

        spectra_path = work_path.with_name(f"{work_path.stem}_spectra.txt")
        table = np.loadtxt(spectra_path, delimiter=",", skiprows=1)
        if table.ndim == 1:
            table = table.reshape(1, -1)

        return {
            "k": np.asarray(table[:, 0], dtype=np.float64).tolist(),
            "e_total": np.asarray(table[:, 1], dtype=np.float64).tolist(),
            "enstrophy": np.asarray(table[:, 4], dtype=np.float64).tolist(),
            "spectral_total_ke": float(np.sum(table[:, 1], dtype=np.float64)),
            "spectral_total_enstrophy": float(np.sum(table[:, 4], dtype=np.float64)),
        }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MPI worker for rank-consistency spectra tests.")
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
