from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mpi4py import MPI

from postprocess_fft.app import analyze_file_parallel


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CASES = {
    "sample_txt": REPO_ROOT / "data" / "SampledData0.txt",
    "dedalus_h5": (
        REPO_ROOT
        / "data"
        / "spectralDNS_tgv_incomp_Re400NumPtsPerDir64"
        / "tgv_out_Re400NumPtsPerDir64_fields"
        / "tgv_out_Re400NumPtsPerDir64_fields_s2_write00007.h5"
    ),
}
SUMMARY_LINES: list[str] = []


def relative_l2_error(values: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(values - reference))
    denominator = float(np.linalg.norm(reference))
    return numerator / max(denominator, 1.0e-30)


def collect_case_metrics(path: Path) -> dict[str, np.ndarray | float] | None:
    with tempfile.TemporaryDirectory(prefix="ke_spectra_test_") as tmpdir:
        work_path = Path(tmpdir) / path.name
        shutil.copy2(path, work_path)

        if MPI.COMM_WORLD.rank == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze_file_parallel(
                    str(work_path),
                    MPI.COMM_WORLD,
                    backend_name="heffte_fftw",
                    visualize=False,
                )
        else:
            result = analyze_file_parallel(
                str(work_path),
                MPI.COMM_WORLD,
                backend_name="heffte_fftw",
                visualize=False,
            )
        MPI.COMM_WORLD.Barrier()

        if MPI.COMM_WORLD.rank != 0:
            return None

        spectra_path = work_path.with_name(f"{work_path.stem}_spectra.txt")
        metadata_path = work_path.with_name(f"{work_path.stem}_spectra_metadata.txt")
        if not spectra_path.exists():
            raise FileNotFoundError(spectra_path)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        table = np.loadtxt(spectra_path, delimiter=",", skiprows=1)
        if table.ndim == 1:
            table = table.reshape(1, -1)

        return {
            "k": np.asarray(table[:, 0], dtype=np.float64),
            "e_total": np.asarray(table[:, 1], dtype=np.float64),
            "enstrophy_spectrum": np.asarray(table[:, 4], dtype=np.float64),
            "spectral_total_ke": float(np.sum(table[:, 1], dtype=np.float64)),
            "spectral_total_enstrophy": float(np.sum(table[:, 4], dtype=np.float64)),
        }


class TestKESpectraComparison(unittest.TestCase):
    def test_spectral_totals_are_finite_and_nonnegative(self) -> None:
        rank = MPI.COMM_WORLD.rank
        for case_name in sorted(DATA_CASES):
            with self.subTest(case=case_name):
                metrics = collect_case_metrics(DATA_CASES[case_name])
                if rank != 0:
                    continue

                assert metrics is not None
                self.assertTrue(np.isfinite(metrics["spectral_total_ke"]))
                self.assertTrue(np.isfinite(metrics["spectral_total_enstrophy"]))
                self.assertGreaterEqual(metrics["spectral_total_ke"], 0.0)
                self.assertGreaterEqual(metrics["spectral_total_enstrophy"], 0.0)

    def test_report_full_spectra_relative_l2_errors(self) -> None:
        sample_metrics = collect_case_metrics(DATA_CASES["sample_txt"])
        dedalus_metrics = collect_case_metrics(DATA_CASES["dedalus_h5"])
        if MPI.COMM_WORLD.rank != 0:
            return

        assert sample_metrics is not None
        assert dedalus_metrics is not None

        common_len = min(len(sample_metrics["k"]), len(dedalus_metrics["k"]))
        sample_e = np.asarray(sample_metrics["e_total"][:common_len], dtype=np.float64)
        dedalus_e = np.asarray(dedalus_metrics["e_total"][:common_len], dtype=np.float64)
        sample_en = np.asarray(sample_metrics["enstrophy_spectrum"][:common_len], dtype=np.float64)
        dedalus_en = np.asarray(dedalus_metrics["enstrophy_spectrum"][:common_len], dtype=np.float64)

        etot_rel_l2 = relative_l2_error(sample_e, dedalus_e)
        enst_rel_l2 = relative_l2_error(sample_en, dedalus_en)

        SUMMARY_LINES.extend(
            [
                "Full-spectrum comparison",
                f"  Common spectrum length:         {common_len:d}",
                f"  Relative L2 error in E_total:   {etot_rel_l2:.16e}",
                f"  Relative L2 error in Enstrophy: {enst_rel_l2:.16e}",
            ]
        )
        self.assertTrue(np.isfinite(etot_rel_l2))
        self.assertTrue(np.isfinite(enst_rel_l2))

    def test_report_txt_vs_dedalus_differences(self) -> None:
        sample_metrics = collect_case_metrics(DATA_CASES["sample_txt"])
        dedalus_metrics = collect_case_metrics(DATA_CASES["dedalus_h5"])
        if MPI.COMM_WORLD.rank != 0:
            return

        assert sample_metrics is not None
        assert dedalus_metrics is not None

        ke_abs_diff = abs(sample_metrics["spectral_total_ke"] - dedalus_metrics["spectral_total_ke"])
        enst_abs_diff = abs(sample_metrics["spectral_total_enstrophy"] - dedalus_metrics["spectral_total_enstrophy"])
        ke_rel_diff = ke_abs_diff / max(abs(dedalus_metrics["spectral_total_ke"]), 1.0e-30)
        enst_rel_diff = enst_abs_diff / max(abs(dedalus_metrics["spectral_total_enstrophy"]), 1.0e-30)

        SUMMARY_LINES.extend(
            [
                "Integrated totals from saved spectra",
                f"  Sample TXT total KE:           {sample_metrics['spectral_total_ke']:.16e}",
                f"  Dedalus HDF5 total KE:         {dedalus_metrics['spectral_total_ke']:.16e}",
                f"  |dKE|:                         {ke_abs_diff:.16e}",
                f"  Relative |dKE|:                {ke_rel_diff:.16e}",
                f"  Sample TXT total enstrophy:    {sample_metrics['spectral_total_enstrophy']:.16e}",
                f"  Dedalus HDF5 total enstrophy:  {dedalus_metrics['spectral_total_enstrophy']:.16e}",
                f"  |dEnstrophy|:                  {enst_abs_diff:.16e}",
                f"  Relative |dEnstrophy|:         {enst_rel_diff:.16e}",
            ]
        )


def run_mpi_test_suite() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestKESpectraComparison)

    if rank == 0:
        stream = sys.stdout
        verbosity = 2
    else:
        stream = open(os.devnull, "w", encoding="utf-8")
        verbosity = 0

    try:
        result = unittest.TextTestRunner(stream=stream, verbosity=verbosity).run(suite)
    finally:
        if rank != 0:
            stream.close()

    local_ok = 1 if result.wasSuccessful() else 0
    global_ok = comm.allreduce(local_ok, op=MPI.MIN)
    comm.Barrier()
    if rank == 0 and SUMMARY_LINES:
        print()
        print("Comparison summary")
        print("-" * 72)
        for line in SUMMARY_LINES:
            print(line)
    return 0 if global_ok else 1


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = run_mpi_test_suite()
    finally:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()
    raise SystemExit(exit_code)
