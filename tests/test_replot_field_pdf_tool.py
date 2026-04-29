from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "replot_field_pdf.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("replot_field_pdf_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("replot_field_pdf_tool", module)
    spec.loader.exec_module(module)
    return module


tool = _load_tool_module()


class TestReplotFieldPdfTool(unittest.TestCase):
    def test_ordered_available_pdf_names_prefers_registry_order(self) -> None:
        pdfs = {
            "normalized_mach_number": {},
            "normalized_pressure": {},
            "normalized_density": {},
            "rms_normalized_u3": {},
            "rms_normalized_u2": {},
            "normalized_u12_by_vorticity_rms": {},
            "normalized_u": {},
            "normalized_vorticity_magnitude": {},
            "normalized_velocity_magnitude": {},
            "normalized_dilatation": {},
        }
        self.assertEqual(
            tool.ordered_available_pdf_names(pdfs),
            [
                "normalized_dilatation",
                "normalized_velocity_magnitude",
                "normalized_vorticity_magnitude",
                "normalized_u",
                "normalized_u12_by_vorticity_rms",
                "rms_normalized_u2",
                "rms_normalized_u3",
                "normalized_density",
                "normalized_pressure",
                "normalized_mach_number",
            ],
        )

    def test_resolve_pdf_selector_accepts_numeric_index(self) -> None:
        pdfs = {
            "normalized_mach_number": {},
            "normalized_pressure": {},
            "normalized_density": {},
            "rms_normalized_u3": {},
            "rms_normalized_u2": {},
            "normalized_u12_by_vorticity_rms": {},
            "normalized_u": {},
            "normalized_vorticity_magnitude": {},
            "normalized_velocity_magnitude": {},
            "normalized_dilatation": {},
        }
        self.assertEqual(tool.resolve_pdf_selector("1", pdfs), "normalized_dilatation")
        self.assertEqual(tool.resolve_pdf_selector("2", pdfs), "normalized_velocity_magnitude")
        self.assertEqual(tool.resolve_pdf_selector("3", pdfs), "normalized_vorticity_magnitude")
        self.assertEqual(tool.resolve_pdf_selector("4", pdfs), "normalized_u")
        self.assertEqual(tool.resolve_pdf_selector("5", pdfs), "normalized_u12_by_vorticity_rms")
        self.assertEqual(tool.resolve_pdf_selector("6", pdfs), "rms_normalized_u2")
        self.assertEqual(tool.resolve_pdf_selector("7", pdfs), "rms_normalized_u3")
        self.assertEqual(tool.resolve_pdf_selector("8", pdfs), "normalized_density")
        self.assertEqual(tool.resolve_pdf_selector("9", pdfs), "normalized_pressure")
        self.assertEqual(tool.resolve_pdf_selector("10", pdfs), "normalized_mach_number")

    def test_resolve_pdf_selector_accepts_aliases(self) -> None:
        pdfs = {
            "normalized_mach_number": {},
            "normalized_pressure": {},
            "normalized_density": {},
            "rms_normalized_u3": {},
            "rms_normalized_u2": {},
            "normalized_u12_by_vorticity_rms": {},
            "normalized_u": {},
            "normalized_vorticity_magnitude": {},
            "normalized_velocity_magnitude": {},
            "normalized_dilatation": {},
        }
        self.assertEqual(tool.resolve_pdf_selector("velocity", pdfs, normalized=True), "normalized_velocity_magnitude")
        self.assertEqual(tool.resolve_pdf_selector("vorticity", pdfs), "normalized_vorticity_magnitude")
        self.assertEqual(tool.resolve_pdf_selector("omega", pdfs, normalized=True), "normalized_vorticity_magnitude")
        self.assertEqual(tool.resolve_pdf_selector("u", pdfs), "normalized_u")
        self.assertEqual(tool.resolve_pdf_selector("vx", pdfs, normalized=True), "normalized_u")
        self.assertEqual(tool.resolve_pdf_selector("u1", pdfs), "normalized_u")
        self.assertEqual(tool.resolve_pdf_selector("u12", pdfs), "normalized_u12_by_vorticity_rms")
        self.assertEqual(tool.resolve_pdf_selector("u1,2", pdfs), "normalized_u12_by_vorticity_rms")
        self.assertEqual(tool.resolve_pdf_selector("dux_dy", pdfs), "normalized_u12_by_vorticity_rms")
        self.assertEqual(tool.resolve_pdf_selector("u2", pdfs), "rms_normalized_u2")
        self.assertEqual(tool.resolve_pdf_selector("v", pdfs), "rms_normalized_u2")
        self.assertEqual(tool.resolve_pdf_selector("u3", pdfs), "rms_normalized_u3")
        self.assertEqual(tool.resolve_pdf_selector("w", pdfs), "rms_normalized_u3")
        self.assertEqual(tool.resolve_pdf_selector("density", pdfs, normalized=True), "normalized_density")
        self.assertEqual(tool.resolve_pdf_selector("normalized_dilation", pdfs), "normalized_dilatation")
        self.assertEqual(tool.resolve_pdf_selector("dilation", pdfs), "normalized_dilatation")
        self.assertEqual(tool.resolve_pdf_selector("mach", pdfs), "normalized_mach_number")

    def test_resolve_pdf_selector_rejects_out_of_range_index(self) -> None:
        pdfs = {
            "normalized_mach_number": {},
            "normalized_pressure": {},
            "normalized_density": {},
            "rms_normalized_u3": {},
            "rms_normalized_u2": {},
            "normalized_u12_by_vorticity_rms": {},
            "normalized_u": {},
            "normalized_vorticity_magnitude": {},
            "normalized_velocity_magnitude": {},
            "normalized_dilatation": {},
        }
        with self.assertRaisesRegex(ValueError, "out of range"):
            tool.resolve_pdf_selector("11", pdfs)

    def test_main_uses_pdf_smooth_output_when_smoothed_is_requested(self) -> None:
        pdf_result = {
            "source_h5": str(REPO_ROOT / "data" / "slice_data" / "sample_slices.h5"),
            "source_field": "density",
            "normalization": "global_std",
            "bin_edges": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "bin_centers": np.array([0.5, 1.5], dtype=np.float64),
            "counts": np.array([1, 3], dtype=np.int64),
            "pdf": np.array([0.25, 0.75], dtype=np.float64),
            "x_label": "x",
            "plot_title": "Density PDF",
            "y_label": "PDF",
            "normalization_scale": 1.0,
            "normalization_offset": 0.0,
        }
        smoothed_result = {
            **pdf_result,
            "smoothing": "gaussian_weighted_bins",
            "smoothing_sigma_bins": 2.0,
        }

        argv = [
            "replot_field_pdf.py",
            "data/slice_data/sample_slices.h5",
            "--pdf",
            "normalized_density",
            "--smoothed",
            "--smooth-sigma-bins",
            "2.0",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(tool, "list_available_pdfs", return_value={"normalized_density": {}}):
                with mock.patch.object(
                    tool,
                    "load_saved_pdf",
                    return_value={
                        "attrs": pdf_result,
                        "bin_edges": pdf_result["bin_edges"],
                        "bin_centers": pdf_result["bin_centers"],
                        "counts": pdf_result["counts"],
                        "pdf": pdf_result["pdf"],
                    },
                ):
                    with mock.patch.object(tool, "smooth_field_pdf_for_plot", return_value=smoothed_result):
                        with mock.patch.object(tool, "plot_smoothed_field_pdf") as plot_smoothed:
                            with mock.patch.object(tool, "plot_field_pdf") as plot_raw:
                                with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                                    tool.main()

        plot_raw.assert_not_called()
        plot_smoothed.assert_called_once()
        output_path = plot_smoothed.call_args.args[1]
        self.assertIn("/slice_plots/pdf_smooth/", output_path)
        self.assertIn("Applied smoothing: gaussian_weighted_bins", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
