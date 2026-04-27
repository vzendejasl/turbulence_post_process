from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI

from postprocess_vis.app import run_visualization
from postprocess_vis.slice_data import load_saved_pdf


def collect_metrics(input_path: Path) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix="field_pdf_integration_worker_") as tmpdir:
        work_path = Path(tmpdir) / input_path.name
        shutil.copy2(input_path, work_path)

        if MPI.COMM_WORLD.rank == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                _, slice_data_path = run_visualization(
                    str(work_path),
                    comm=MPI.COMM_WORLD,
                    assume_structured_h5=True,
                    save_slice_data=True,
                    pdf_only=True,
                    pdf_bins=32,
                )
        else:
            _, slice_data_path = run_visualization(
                str(work_path),
                comm=MPI.COMM_WORLD,
                assume_structured_h5=True,
                save_slice_data=True,
                pdf_only=True,
                pdf_bins=32,
            )
        MPI.COMM_WORLD.Barrier()

        if MPI.COMM_WORLD.rank != 0:
            return None

        if slice_data_path is None:
            raise RuntimeError("Expected a slice-data HDF5 output path from pdf-only visualization.")
        loaded = load_saved_pdf(slice_data_path, "normalized_dilatation")
        attrs = loaded["attrs"]
        return {
            "bin_edges": np.asarray(loaded["bin_edges"], dtype=np.float64).tolist(),
            "counts": np.asarray(loaded["counts"], dtype=np.int64).tolist(),
            "pdf": np.asarray(loaded["pdf"], dtype=np.float64).tolist(),
            "source_field": str(attrs["source_field"]),
            "normalization": str(attrs["normalization"]),
            "normalization_scale": float(attrs["normalization_scale"]),
            "measured_normalization_scale": float(attrs.get("measured_normalization_scale", attrs["normalization_scale"])),
            "value_range_min": float(attrs["value_range_min"]),
            "value_range_max": float(attrs["value_range_max"]),
            "total_samples": int(attrs["total_samples"]),
            "in_range_samples": int(attrs["in_range_samples"]),
            "underflow_count": int(attrs["underflow_count"]),
            "overflow_count": int(attrs["overflow_count"]),
            "pdf_integral": float(attrs["pdf_integral"]),
        }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MPI worker for full field-PDF integration rank-consistency tests.")
    parser.add_argument("input_path", help="Structured HDF5 input path")
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
