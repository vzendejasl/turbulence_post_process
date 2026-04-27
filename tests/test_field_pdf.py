from __future__ import annotations

import unittest

import numpy as np

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - environment dependent
    MPI = None

from postprocess_vis.pdfs import compute_distributed_field_pdf


@unittest.skipIf(MPI is None, "mpi4py is not installed in this test environment")
class TestFieldPdf(unittest.TestCase):
    def test_pdf_integrates_to_one(self) -> None:
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        result = compute_distributed_field_pdf(
            values,
            MPI.COMM_SELF,
            bins=5,
            normalization_scale=1.0,
            pdf_name="test_pdf",
            source_field="test_field",
        )

        self.assertEqual(int(result["total_samples"]), 5)
        self.assertEqual(int(np.sum(result["counts"], dtype=np.int64)), 5)
        self.assertAlmostEqual(float(result["pdf_integral"]), 1.0, places=12)

    def test_scaling_invariance_under_rms_normalization(self) -> None:
        values = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        rms = float(np.sqrt(np.mean(values**2)))
        scaled_values = 7.5 * values
        scaled_rms = float(np.sqrt(np.mean(scaled_values**2)))

        result = compute_distributed_field_pdf(
            values,
            MPI.COMM_SELF,
            bins=8,
            normalization_scale=rms,
            pdf_name="test_pdf",
            source_field="test_field",
            value_range=(-2.0, 2.0),
        )
        scaled_result = compute_distributed_field_pdf(
            scaled_values,
            MPI.COMM_SELF,
            bins=8,
            normalization_scale=scaled_rms,
            pdf_name="test_pdf",
            source_field="test_field",
            value_range=(-2.0, 2.0),
        )

        np.testing.assert_allclose(result["counts"], scaled_result["counts"])
        np.testing.assert_allclose(result["pdf"], scaled_result["pdf"])
        np.testing.assert_allclose(result["bin_edges"], scaled_result["bin_edges"])

    def test_zero_rms_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalization_scale"):
            compute_distributed_field_pdf(
                np.zeros(8, dtype=np.float64),
                MPI.COMM_SELF,
                bins=8,
                normalization_scale=0.0,
                pdf_name="test_pdf",
                source_field="test_field",
            )

    def test_fixed_range_reports_underflow_and_overflow(self) -> None:
        values = np.array([-5.0, -1.0, 0.0, 1.0, 6.0], dtype=np.float64)
        result = compute_distributed_field_pdf(
            values,
            MPI.COMM_SELF,
            bins=4,
            value_range=(-2.0, 2.0),
            normalization_scale=1.0,
            pdf_name="test_pdf",
            source_field="test_field",
        )

        self.assertEqual(int(result["underflow_count"]), 1)
        self.assertEqual(int(result["overflow_count"]), 1)
        self.assertIn("Widen the range", str(result["range_warning"]))


if __name__ == "__main__":
    unittest.main()
