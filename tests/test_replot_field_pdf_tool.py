from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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
            "normalized_u": {},
            "normalized_vorticity_magnitude": {},
            "normalized_velocity_magnitude": {},
            "normalized_dilatation": {},
        }
        self.assertEqual(tool.resolve_pdf_selector("1", pdfs), "normalized_dilatation")
        self.assertEqual(tool.resolve_pdf_selector("2", pdfs), "normalized_velocity_magnitude")
        self.assertEqual(tool.resolve_pdf_selector("3", pdfs), "normalized_vorticity_magnitude")
        self.assertEqual(tool.resolve_pdf_selector("4", pdfs), "normalized_u")
        self.assertEqual(tool.resolve_pdf_selector("5", pdfs), "normalized_density")
        self.assertEqual(tool.resolve_pdf_selector("6", pdfs), "normalized_pressure")
        self.assertEqual(tool.resolve_pdf_selector("7", pdfs), "normalized_mach_number")

    def test_resolve_pdf_selector_accepts_aliases(self) -> None:
        pdfs = {
            "normalized_mach_number": {},
            "normalized_pressure": {},
            "normalized_density": {},
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
        self.assertEqual(tool.resolve_pdf_selector("density", pdfs, normalized=True), "normalized_density")
        self.assertEqual(tool.resolve_pdf_selector("normalized_dilation", pdfs), "normalized_dilatation")
        self.assertEqual(tool.resolve_pdf_selector("dilation", pdfs), "normalized_dilatation")
        self.assertEqual(tool.resolve_pdf_selector("mach", pdfs), "normalized_mach_number")

    def test_resolve_pdf_selector_rejects_out_of_range_index(self) -> None:
        pdfs = {
            "normalized_mach_number": {},
            "normalized_pressure": {},
            "normalized_density": {},
            "normalized_u": {},
            "normalized_vorticity_magnitude": {},
            "normalized_velocity_magnitude": {},
            "normalized_dilatation": {},
        }
        with self.assertRaisesRegex(ValueError, "out of range"):
            tool.resolve_pdf_selector("8", pdfs)


if __name__ == "__main__":
    unittest.main()
