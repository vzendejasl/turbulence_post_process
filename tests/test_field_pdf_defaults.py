from __future__ import annotations

import os
import tempfile
import unittest

import h5py
import numpy as np

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - environment dependent
    MPI = None

from postprocess_vis.app import run_visualization
from postprocess_vis.pdfs import default_field_pdf_specs
from postprocess_vis.slice_data import list_available_pdfs


class TestFieldPdfDefaults(unittest.TestCase):
    def test_default_field_pdf_specs_include_density_and_pressure_when_available(self) -> None:
        field_specs = [
            ("div_u", "div_u", r"$\theta$", "divergence"),
            ("velocity_magnitude", "velocity_magnitude", r"$|\mathbf{u}|$", "velocity"),
            ("dux_dy", "dux_dy", r"$u_{1,2}$", "velocity_gradient"),
            ("vorticity_magnitude", "vorticity_magnitude", r"$|\boldsymbol{\omega}|$", "vorticity"),
            ("vx", "vx", r"$u_1$", "velocity"),
            ("vy", "vy", r"$u_2$", "velocity"),
            ("vz", "vz", r"$u_3$", "velocity"),
            ("density", "density", r"$\rho$", "scalar"),
            ("pressure", "pressure", r"$p$", "scalar"),
            ("mach_number", "mach_number", r"$M$", "thermo"),
        ]

        specs = default_field_pdf_specs(field_specs)
        self.assertEqual(
            [spec["pdf_name"] for spec in specs],
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

    @unittest.skipIf(MPI is None, "mpi4py is not installed in this test environment")
    def test_pdf_only_visualization_stores_velocity_density_pressure_and_mach_pdfs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="field_pdf_defaults_") as tmpdir:
            input_path = os.path.join(tmpdir, "sample.h5")
            with h5py.File(input_path, "w") as hf:
                hf.attrs["step"] = "0"
                hf.attrs["time"] = 0.0
                hf.attrs["periodic_duplicate_last"] = False
                grid = hf.create_group("grid")
                for axis_name in ("x", "y", "z"):
                    grid.create_dataset(axis_name, data=np.array([0.0, 1.0], dtype=np.float64))
                fields = hf.create_group("fields")
                shape = (2, 2, 2)
                fields.create_dataset("vx", data=np.zeros(shape, dtype=np.float64))
                fields.create_dataset("vy", data=np.zeros(shape, dtype=np.float64))
                fields.create_dataset("vz", data=np.zeros(shape, dtype=np.float64))
                density = fields.create_dataset(
                    "density",
                    data=np.array(
                        [[[1.0, 1.1], [1.2, 1.3]], [[1.4, 1.5], [1.6, 1.7]]],
                        dtype=np.float64,
                    ),
                )
                density.attrs["display_name"] = "Density"
                density.attrs["plot_label"] = r"$\rho$"
                pressure = fields.create_dataset(
                    "pressure",
                    data=np.array(
                        [[[2.0, 2.2], [2.4, 2.6]], [[2.8, 3.0], [3.2, 3.4]]],
                        dtype=np.float64,
                    ),
                )
                pressure.attrs["display_name"] = "Pressure"
                pressure.attrs["plot_label"] = r"$p$"

            _, slice_data_path = run_visualization(
                input_path,
                comm=MPI.COMM_SELF,
                assume_structured_h5=True,
                save_slice_data=True,
                pdf_only=True,
                pdf_bins=16,
            )

            self.assertIsNotNone(slice_data_path)

            available = list_available_pdfs(slice_data_path)
            self.assertIn("normalized_dilatation", available)
            self.assertIn("normalized_velocity_magnitude", available)
            self.assertIn("normalized_vorticity_magnitude", available)
            self.assertIn("normalized_u", available)
            self.assertIn("normalized_u12_by_vorticity_rms", available)
            self.assertIn("rms_normalized_u2", available)
            self.assertIn("rms_normalized_u3", available)
            self.assertIn("normalized_density", available)
            self.assertIn("normalized_pressure", available)
            self.assertIn("normalized_mach_number", available)
            self.assertEqual(available["normalized_velocity_magnitude"]["source_field"], "velocity_magnitude")
            self.assertEqual(available["normalized_velocity_magnitude"]["normalization"], "global_std")
            self.assertEqual(available["normalized_vorticity_magnitude"]["source_field"], "vorticity_magnitude")
            self.assertEqual(available["normalized_vorticity_magnitude"]["normalization"], "global_std")
            self.assertEqual(available["normalized_u"]["source_field"], "vx")
            self.assertEqual(available["normalized_u"]["normalization"], "global_std")
            self.assertEqual(available["normalized_u12_by_vorticity_rms"]["source_field"], "dux_dy")
            self.assertEqual(
                available["normalized_u12_by_vorticity_rms"]["normalization"],
                "reference_global_rms",
            )
            self.assertEqual(
                available["normalized_u12_by_vorticity_rms"]["normalization_reference_field"],
                "vorticity_magnitude",
            )
            self.assertEqual(available["rms_normalized_u2"]["source_field"], "vy")
            self.assertEqual(available["rms_normalized_u2"]["normalization"], "global_rms")
            self.assertEqual(available["rms_normalized_u3"]["source_field"], "vz")
            self.assertEqual(available["rms_normalized_u3"]["normalization"], "global_rms")
            self.assertEqual(available["normalized_density"]["source_field"], "density")
            self.assertEqual(available["normalized_density"]["normalization"], "global_std")
            self.assertEqual(available["normalized_pressure"]["source_field"], "pressure")
            self.assertEqual(available["normalized_pressure"]["normalization"], "global_std")
            self.assertEqual(available["normalized_mach_number"]["source_field"], "mach_number")
            self.assertEqual(available["normalized_mach_number"]["normalization"], "global_std")

    @unittest.skipIf(MPI is None, "mpi4py is not installed in this test environment")
    def test_pdf_only_visualization_with_velocity_only_input_skips_missing_thermo_pdfs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="field_pdf_velocity_only_") as tmpdir:
            input_path = os.path.join(tmpdir, "velocity_only.h5")
            with h5py.File(input_path, "w") as hf:
                hf.attrs["step"] = "0"
                hf.attrs["time"] = 0.0
                hf.attrs["periodic_duplicate_last"] = False
                grid = hf.create_group("grid")
                for axis_name in ("x", "y", "z"):
                    grid.create_dataset(axis_name, data=np.array([0.0, 1.0], dtype=np.float64))
                fields = hf.create_group("fields")
                shape = (2, 2, 2)
                fields.create_dataset(
                    "vx",
                    data=np.array(
                        [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
                        dtype=np.float64,
                    ),
                )
                fields.create_dataset("vy", data=np.zeros(shape, dtype=np.float64))
                fields.create_dataset("vz", data=np.zeros(shape, dtype=np.float64))

            _, slice_data_path = run_visualization(
                input_path,
                comm=MPI.COMM_SELF,
                assume_structured_h5=True,
                save_slice_data=True,
                pdf_only=True,
                pdf_bins=16,
            )

            self.assertIsNotNone(slice_data_path)

            available = list_available_pdfs(slice_data_path)
            self.assertIn("normalized_dilatation", available)
            self.assertIn("normalized_velocity_magnitude", available)
            self.assertIn("normalized_vorticity_magnitude", available)
            self.assertIn("normalized_u", available)
            self.assertIn("normalized_u12_by_vorticity_rms", available)
            self.assertIn("rms_normalized_u2", available)
            self.assertIn("rms_normalized_u3", available)
            self.assertNotIn("normalized_density", available)
            self.assertNotIn("normalized_pressure", available)
            self.assertNotIn("normalized_mach_number", available)
            self.assertEqual(available["normalized_velocity_magnitude"]["source_field"], "velocity_magnitude")
            self.assertEqual(available["normalized_vorticity_magnitude"]["source_field"], "vorticity_magnitude")
            self.assertEqual(available["normalized_u"]["source_field"], "vx")
            self.assertEqual(available["normalized_u12_by_vorticity_rms"]["source_field"], "dux_dy")
            self.assertEqual(
                available["normalized_u12_by_vorticity_rms"]["normalization_reference_field"],
                "vorticity_magnitude",
            )
            self.assertEqual(available["rms_normalized_u2"]["normalization"], "global_rms")
            self.assertEqual(available["rms_normalized_u3"]["normalization"], "global_rms")


if __name__ == "__main__":
    unittest.main()
