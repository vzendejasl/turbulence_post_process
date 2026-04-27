from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CASE = REPO_ROOT / "data" / "SampledData0.h5"
RANKS = (1, 2, 4)
REL_TOL = 1.0e-12


def run_worker(case_path: Path, ranks: int, output_json: Path) -> dict[str, object]:
    command = [
        "mpirun",
        "-n",
        str(ranks),
        sys.executable,
        "-m",
        "tests.field_pdf_integration_worker",
        str(case_path),
        str(output_json),
    ]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Field-PDF integration worker failed for {case_path.name} at {ranks} rank(s).\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    with open(output_json, "r", encoding="utf-8") as handle:
        return json.load(handle)


@unittest.skipIf(shutil.which("mpirun") is None, "mpirun is not installed in this test environment")
class TestFieldPdfIntegrationRankConsistency(unittest.TestCase):
    def test_tgv_pdf_is_rank_consistent_and_divergence_is_roundoff(self) -> None:
        with tempfile.TemporaryDirectory(prefix="field_pdf_integration_rank_consistency_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            results = {
                ranks: run_worker(DATA_CASE, ranks, tmpdir_path / f"field_pdf_integration_r{ranks}.json")
                for ranks in RANKS
            }

        reference = results[1]
        self.assertEqual(reference["source_field"], "div_u")
        self.assertEqual(reference["normalization"], "global_rms")
        self.assertEqual(int(reference["total_samples"]), 64 * 64 * 64)
        self.assertEqual(int(reference["underflow_count"]), 0)
        self.assertEqual(int(reference["overflow_count"]), 0)
        self.assertAlmostEqual(float(reference["pdf_integral"]), 1.0, places=12)
        self.assertEqual(float(reference["normalization_scale"]), 0.0)
        self.assertLess(float(reference["measured_normalization_scale"]), 1.0e-10)

        reference_edges = np.asarray(reference["bin_edges"], dtype=np.float64)
        reference_counts = np.asarray(reference["counts"], dtype=np.int64)
        reference_pdf = np.asarray(reference["pdf"], dtype=np.float64)

        for ranks in RANKS[1:]:
            trial = results[ranks]
            np.testing.assert_allclose(np.asarray(trial["bin_edges"], dtype=np.float64), reference_edges, rtol=0.0, atol=0.0)
            np.testing.assert_array_equal(np.asarray(trial["counts"], dtype=np.int64), reference_counts)
            np.testing.assert_allclose(np.asarray(trial["pdf"], dtype=np.float64), reference_pdf, rtol=REL_TOL, atol=0.0)
            self.assertEqual(float(trial["normalization_scale"]), 0.0)
            self.assertLess(float(trial["measured_normalization_scale"]), 1.0e-10)
            self.assertAlmostEqual(float(trial["value_range_min"]), float(reference["value_range_min"]), places=20)
            self.assertAlmostEqual(float(trial["value_range_max"]), float(reference["value_range_max"]), places=20)
            self.assertEqual(int(trial["total_samples"]), int(reference["total_samples"]))
            self.assertEqual(int(trial["in_range_samples"]), int(reference["in_range_samples"]))
            self.assertEqual(int(trial["underflow_count"]), 0)
            self.assertEqual(int(trial["overflow_count"]), 0)
            self.assertAlmostEqual(float(trial["pdf_integral"]), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
