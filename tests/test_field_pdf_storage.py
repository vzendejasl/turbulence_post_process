from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from postprocess_vis.slice_data import initialize_slice_data_file
from postprocess_vis.slice_data import list_available_pdfs
from postprocess_vis.slice_data import load_saved_pdf
from postprocess_vis.slice_data import save_pdf_serial


class TestFieldPdfStorage(unittest.TestCase):
    def test_save_and_load_pdf(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        handle.close()
        self.addCleanup(lambda: os.unlink(handle.name))

        meta = {
            "step": "0",
            "time": 0.0,
            "shape": (2, 2, 2),
            "x": np.array([0.0, 1.0], dtype=np.float64),
            "y": np.array([0.0, 1.0], dtype=np.float64),
            "z": np.array([0.0, 1.0], dtype=np.float64),
        }
        initialize_slice_data_file(
            handle.name,
            meta,
            [("div_u", "div_u", r"$\nabla \cdot \mathbf{u}$", "divergence")],
            [("z", 0, "xy_center")],
            source_file=handle.name,
            source_h5=handle.name,
            backend_name="heffte_fftw",
        )

        pdf_result = {
            "pdf_name": "normalized_dilatation",
            "source_field": "div_u",
            "normalization": "global_std",
            "plot_title": "Normalized Dilatation PDF",
            "x_label": "x",
            "y_label": "PDF",
            "bin_edges": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
            "bin_centers": np.array([-0.5, 0.5], dtype=np.float64),
            "counts": np.array([2, 2], dtype=np.int64),
            "pdf": np.array([0.5, 0.5], dtype=np.float64),
            "total_samples": 4,
            "in_range_samples": 4,
            "underflow_count": 0,
            "overflow_count": 0,
            "source_field_min": -4.0,
            "source_field_max": 4.0,
            "source_field_mean": 0.0,
            "source_field_std": 2.0,
            "source_field_rms": 2.0,
            "normalization_scale": 2.0,
            "normalization_offset": 0.0,
            "value_range_min": -1.0,
            "value_range_max": 1.0,
            "bin_count": 2,
            "range_mode": "variable",
            "pdf_integral": 1.0,
            "binning_warning": "warning",
            "range_warning": "",
        }
        save_pdf_serial(handle.name, "normalized_dilatation", pdf_result)

        available = list_available_pdfs(handle.name)
        self.assertIn("normalized_dilatation", available)

        loaded = load_saved_pdf(handle.name, "normalized_dilatation")
        np.testing.assert_allclose(loaded["bin_edges"], pdf_result["bin_edges"])
        np.testing.assert_allclose(loaded["pdf"], pdf_result["pdf"])
        self.assertEqual(str(loaded["attrs"]["source_field"]), "div_u")
        self.assertEqual(str(loaded["attrs"]["normalization"]), "global_std")
        self.assertEqual(float(loaded["attrs"]["source_field_min"]), -4.0)
        self.assertEqual(float(loaded["attrs"]["source_field_max"]), 4.0)
        self.assertEqual(float(loaded["attrs"]["source_field_std"]), 2.0)
        self.assertEqual(float(loaded["attrs"]["source_field_rms"]), 2.0)
        self.assertEqual(float(loaded["attrs"]["normalization_offset"]), 0.0)


if __name__ == "__main__":
    unittest.main()
