from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import shutil

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RANKS = (1, 2, 4)


def run_worker(ranks: int, output_json: Path) -> dict[str, object]:
    command = [
        "mpirun",
        "-n",
        str(ranks),
        sys.executable,
        "-m",
        "tests.field_pdf_rank_consistency_worker",
        str(output_json),
    ]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        print(completed.stdout)
    if completed.stderr.strip():
        print(completed.stderr)
    with open(output_json, "r", encoding="utf-8") as handle:
        return json.load(handle)


@unittest.skipIf(shutil.which("mpirun") is None, "mpirun is not installed in this test environment")
class TestFieldPdfRankConsistency(unittest.TestCase):
    def test_rank_consistency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="field_pdf_rank_consistency_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            results = {
                ranks: run_worker(ranks, tmpdir_path / f"field_pdf_r{ranks}.json")
                for ranks in RANKS
            }

        reference = results[1]
        reference_edges = np.asarray(reference["bin_edges"], dtype=np.float64)
        reference_counts = np.asarray(reference["counts"], dtype=np.int64)
        reference_pdf = np.asarray(reference["pdf"], dtype=np.float64)

        for ranks in RANKS[1:]:
            trial = results[ranks]
            np.testing.assert_allclose(np.asarray(trial["bin_edges"], dtype=np.float64), reference_edges)
            np.testing.assert_array_equal(np.asarray(trial["counts"], dtype=np.int64), reference_counts)
            np.testing.assert_allclose(np.asarray(trial["pdf"], dtype=np.float64), reference_pdf)
            self.assertEqual(int(trial["total_samples"]), int(reference["total_samples"]))
            self.assertAlmostEqual(float(trial["pdf_integral"]), float(reference["pdf_integral"]), places=12)
            self.assertAlmostEqual(
                float(trial["normalization_scale"]),
                float(reference["normalization_scale"]),
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
