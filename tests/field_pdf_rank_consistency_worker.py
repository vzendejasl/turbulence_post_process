from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from postprocess_vis.pdfs import compute_distributed_field_pdf


def _local_chunk(values):
    chunks = np.array_split(values, MPI.COMM_WORLD.size)
    return np.asarray(chunks[MPI.COMM_WORLD.rank], dtype=np.float64)


def collect_metrics() -> dict[str, object] | None:
    values = np.linspace(-4.0, 6.0, 4096, dtype=np.float64)
    local_values = _local_chunk(values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    result = compute_distributed_field_pdf(
        local_values,
        MPI.COMM_WORLD,
        bins=64,
        normalization_scale=std,
        normalization_offset=mean,
        pdf_name="normalized_dilatation",
        source_field="div_u",
        normalization="global_std",
        plot_title="Normalized Dilatation PDF",
    )
    if MPI.COMM_WORLD.rank != 0:
        return None
    return {
        "bin_edges": np.asarray(result["bin_edges"], dtype=np.float64).tolist(),
        "counts": np.asarray(result["counts"], dtype=np.int64).tolist(),
        "pdf": np.asarray(result["pdf"], dtype=np.float64).tolist(),
        "total_samples": int(result["total_samples"]),
        "pdf_integral": float(result["pdf_integral"]),
        "normalization_scale": float(result["normalization_scale"]),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MPI worker for field-PDF rank consistency tests.")
    parser.add_argument("output_json", help="Rank-0 JSON output path")
    args = parser.parse_args()

    metrics = collect_metrics()
    if MPI.COMM_WORLD.rank == 0:
        with open(Path(args.output_json), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
    MPI.COMM_WORLD.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
