from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from postprocess_lib.prepare import resolve_existing_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CASES = {
    "sample_txt": Path(resolve_existing_path(str(REPO_ROOT / "data" / "SampledData0.txt"))),
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
FULL_SPECTRUM_REL_TOL = 1.0e-12
INTEGRATED_TOTAL_REL_TOL = 1.0e-12


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
        "tests.rank_consistency_worker",
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
            f"Rank-consistency worker failed for {case_path.name} at {ranks} rank(s).\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    with open(output_json, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_rank_family(case_name: str, case_path: Path) -> list[str]:
    lines = [f"Case: {case_name}"]
    with tempfile.TemporaryDirectory(prefix=f"rank_consistency_{case_name}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        metrics_by_rank = {
            ranks: run_worker(case_path, ranks, tmpdir_path / f"{case_name}_r{ranks}.json")
            for ranks in RANKS
        }

    reference = metrics_by_rank[1]
    reference_e = np.asarray(reference["e_total"], dtype=np.float64)
    reference_en = np.asarray(reference["enstrophy"], dtype=np.float64)
    reference_ke = float(reference["spectral_total_ke"])
    reference_enst = float(reference["spectral_total_enstrophy"])

    lines.append(f"  Reference rank count: {RANKS[0]}")
    lines.append(f"  Spectrum length:       {len(reference_e)}")
    lines.append(f"  Relative L2 tolerance: {FULL_SPECTRUM_REL_TOL:.1e}")
    lines.append(f"  Relative sum tolerance:{INTEGRATED_TOTAL_REL_TOL:.1e}")

    for ranks in RANKS[1:]:
        trial = metrics_by_rank[ranks]
        trial_e = np.asarray(trial["e_total"], dtype=np.float64)
        trial_en = np.asarray(trial["enstrophy"], dtype=np.float64)
        common_len = min(len(reference_e), len(trial_e))
        e_rel = relative_l2_error(trial_e[:common_len], reference_e[:common_len])
        en_rel = relative_l2_error(trial_en[:common_len], reference_en[:common_len])
        ke_abs = abs(float(trial["spectral_total_ke"]) - reference_ke)
        enst_abs = abs(float(trial["spectral_total_enstrophy"]) - reference_enst)
        ke_rel = ke_abs / max(abs(reference_ke), 1.0e-30)
        enst_rel = enst_abs / max(abs(reference_enst), 1.0e-30)

        lines.extend(
            [
                f"  Compared against rank {ranks}:",
                f"    Relative L2 error in E_total(k):   {e_rel:.16e}",
                f"    Relative L2 error in Enstrophy(k): {en_rel:.16e}",
                f"    Relative |dKE| from spectra sum:   {ke_rel:.16e}",
                f"    Relative |dEnstrophy| from sum:    {enst_rel:.16e}",
            ]
        )

        if (
            e_rel > FULL_SPECTRUM_REL_TOL
            or en_rel > FULL_SPECTRUM_REL_TOL
            or ke_rel > INTEGRATED_TOTAL_REL_TOL
            or enst_rel > INTEGRATED_TOTAL_REL_TOL
        ):
            raise AssertionError(
                f"Rank consistency failed for {case_name} at {ranks} ranks: "
                f"E_total_rel_l2={e_rel:.16e}, "
                f"Enstrophy_rel_l2={en_rel:.16e}, "
                f"KE_rel={ke_rel:.16e}, "
                f"Enstrophy_rel={enst_rel:.16e}"
            )

    return lines


class TestRankConsistency(unittest.TestCase):
    def test_sample_txt_rank_consistency(self) -> None:
        SUMMARY_LINES.extend(compare_rank_family("sample_txt", DATA_CASES["sample_txt"]))

    def test_dedalus_h5_rank_consistency(self) -> None:
        SUMMARY_LINES.extend(compare_rank_family("dedalus_h5", DATA_CASES["dedalus_h5"]))


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    if SUMMARY_LINES:
        print()
        print("Rank-consistency summary")
        print("-" * 72)
        for line in SUMMARY_LINES:
            print(line)
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
