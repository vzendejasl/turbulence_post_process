from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CASES = {
    "dedalus_h5": (
        REPO_ROOT
        / "data"
        / "spectralDNS_tgv_incomp_Re400NumPtsPerDir64"
        / "tgv_out_Re400NumPtsPerDir64_fields"
        / "tgv_out_Re400NumPtsPerDir64_fields_s2_write00007.h5"
    ),
}
RANKS = (1, 2, 4)
SUMMARY_LINES: list[str] = []
REL_TOL = 1.0e-12


def relative_l2_error(values: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(values - reference))
    denominator = float(np.linalg.norm(reference))
    return numerator / max(denominator, 1.0e-30)


def run_worker(case_path: Path, ranks: int, output_json: Path) -> dict[str, object]:
    command = [
        "mpirun",
        "-n",
        str(ranks),
        sys.executable,
        "-m",
        "tests.third_order_rank_consistency_worker",
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
            f"Third-order rank-consistency worker failed for {case_path.name} at {ranks} rank(s).\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    with open(output_json, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_rank_family(case_name: str, case_path: Path) -> list[str]:
    lines = [f"Case: {case_name}"]
    with tempfile.TemporaryDirectory(prefix=f"third_order_rank_consistency_{case_name}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        metrics_by_rank = {
            ranks: run_worker(case_path, ranks, tmpdir_path / f"{case_name}_r{ranks}.json")
            for ranks in RANKS
        }

    reference = metrics_by_rank[1]
    if int(reference["num_columns"]) != 5:
        raise AssertionError(f"Expected 5 columns in third-order file, got {reference['num_columns']}")

    reference_r = np.asarray(reference["r"], dtype=np.float64)
    reference_s3_x = np.asarray(reference["s3_x"], dtype=np.float64)
    reference_s3_y = np.asarray(reference["s3_y"], dtype=np.float64)
    reference_s3_z = np.asarray(reference["s3_z"], dtype=np.float64)
    reference_s3_avg = np.asarray(reference["s3_avg"], dtype=np.float64)

    lines.append(f"  Reference rank count: {RANKS[0]}")
    lines.append(f"  Third-order length:    {len(reference_r)}")
    lines.append(f"  Relative L2 tolerance: {REL_TOL:.1e}")

    for ranks in RANKS[1:]:
        trial = metrics_by_rank[ranks]
        if int(trial["num_columns"]) != 5:
            raise AssertionError(f"Expected 5 columns in third-order file, got {trial['num_columns']}")

        trial_r = np.asarray(trial["r"], dtype=np.float64)
        trial_s3_x = np.asarray(trial["s3_x"], dtype=np.float64)
        trial_s3_y = np.asarray(trial["s3_y"], dtype=np.float64)
        trial_s3_z = np.asarray(trial["s3_z"], dtype=np.float64)
        trial_s3_avg = np.asarray(trial["s3_avg"], dtype=np.float64)

        common_len = min(len(reference_r), len(trial_r))
        r_rel = relative_l2_error(trial_r[:common_len], reference_r[:common_len])
        s3_x_rel = relative_l2_error(trial_s3_x[:common_len], reference_s3_x[:common_len])
        s3_y_rel = relative_l2_error(trial_s3_y[:common_len], reference_s3_y[:common_len])
        s3_z_rel = relative_l2_error(trial_s3_z[:common_len], reference_s3_z[:common_len])
        s3_avg_rel = relative_l2_error(trial_s3_avg[:common_len], reference_s3_avg[:common_len])

        lines.extend(
            [
                f"  Compared against rank {ranks}:",
                f"    Relative L2 error in r:      {r_rel:.16e}",
                f"    Relative L2 error in S3_x:   {s3_x_rel:.16e}",
                f"    Relative L2 error in S3_y:   {s3_y_rel:.16e}",
                f"    Relative L2 error in S3_z:   {s3_z_rel:.16e}",
                f"    Relative L2 error in S3_avg: {s3_avg_rel:.16e}",
            ]
        )

        if any(value > REL_TOL for value in (r_rel, s3_x_rel, s3_y_rel, s3_z_rel, s3_avg_rel)):
            raise AssertionError(
                f"Third-order rank consistency failed for {case_name} at {ranks} ranks: "
                f"r_rel={r_rel:.16e}, "
                f"S3_x_rel={s3_x_rel:.16e}, "
                f"S3_y_rel={s3_y_rel:.16e}, "
                f"S3_z_rel={s3_z_rel:.16e}, "
                f"S3_avg_rel={s3_avg_rel:.16e}"
            )

    return lines


class TestThirdOrderRankConsistency(unittest.TestCase):
    def test_dedalus_h5_rank_consistency(self) -> None:
        SUMMARY_LINES.extend(compare_rank_family("dedalus_h5", DATA_CASES["dedalus_h5"]))


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    if SUMMARY_LINES:
        print()
        print("Third-order rank-consistency summary")
        print("-" * 72)
        for line in SUMMARY_LINES:
            print(line)
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
