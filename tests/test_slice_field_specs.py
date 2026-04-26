from __future__ import annotations

import os
import tempfile
import unittest

import h5py
import numpy as np

from postprocess_vis.field_specs import build_available_field_specs
from postprocess_vis.field_specs import default_requested_field_names
from postprocess_vis.field_specs import finalize_requested_field_names


class TestSliceFieldSpecs(unittest.TestCase):
    def _write_fields_file(self, include_density: bool, include_pressure: bool) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        handle.close()
        with h5py.File(handle.name, "w") as hf:
            fields = hf.create_group("fields")
            for dataset_name in ("vx", "vy", "vz"):
                fields.create_dataset(dataset_name, data=np.zeros((2, 2, 2), dtype=np.float64))
            if include_density:
                density = fields.create_dataset("density", data=np.ones((2, 2, 2), dtype=np.float64))
                density.attrs["display_name"] = "Density"
                density.attrs["plot_label"] = r"$\rho$"
            if include_pressure:
                pressure = fields.create_dataset("pressure", data=np.ones((2, 2, 2), dtype=np.float64))
                pressure.attrs["display_name"] = "Pressure"
                pressure.attrs["plot_label"] = r"$p$"
        return handle.name

    def test_density_adds_density_gradient_field(self) -> None:
        path = self._write_fields_file(include_density=True, include_pressure=True)
        self.addCleanup(lambda: os.unlink(path))

        with h5py.File(path, "r") as hf:
            field_lookup = build_available_field_specs(hf["fields"])

        self.assertIn("density", field_lookup)
        self.assertIn("pressure", field_lookup)
        self.assertIn("div_u", field_lookup)
        self.assertEqual(field_lookup["div_u"][3], "divergence")
        self.assertIn("sound_speed", field_lookup)
        self.assertIn("mach_number", field_lookup)
        self.assertIn("turbulent_mach_number", field_lookup)
        self.assertIn("density_gradient_magnitude", field_lookup)
        self.assertEqual(field_lookup["density_gradient_magnitude"][3], "density_gradient")

        requested_fields = default_requested_field_names(field_lookup)
        self.assertEqual(
            requested_fields[:6],
            [
                "velocity_magnitude",
                "vorticity_magnitude",
                "div_u",
                "q_criterion",
                "r_criterion",
                "density_gradient_magnitude",
            ],
        )
        self.assertIn("density", requested_fields)
        self.assertIn("pressure", requested_fields)

    def test_density_gradient_not_registered_without_density(self) -> None:
        path = self._write_fields_file(include_density=False, include_pressure=False)
        self.addCleanup(lambda: os.unlink(path))

        with h5py.File(path, "r") as hf:
            field_lookup = build_available_field_specs(hf["fields"])

        self.assertNotIn("density_gradient_magnitude", field_lookup)
        self.assertNotIn("density_gradient_magnitude", default_requested_field_names(field_lookup))
        self.assertNotIn("sound_speed", field_lookup)
        self.assertNotIn("mach_number", field_lookup)
        self.assertNotIn("turbulent_mach_number", field_lookup)

    def test_pressure_without_density_does_not_register_thermo_fields(self) -> None:
        path = self._write_fields_file(include_density=False, include_pressure=True)
        self.addCleanup(lambda: os.unlink(path))

        with h5py.File(path, "r") as hf:
            field_lookup = build_available_field_specs(hf["fields"])

        self.assertNotIn("sound_speed", field_lookup)
        self.assertNotIn("mach_number", field_lookup)
        self.assertNotIn("turbulent_mach_number", field_lookup)

    def test_explicit_requested_fields_still_include_density_gradient(self) -> None:
        path = self._write_fields_file(include_density=True, include_pressure=True)
        self.addCleanup(lambda: os.unlink(path))

        with h5py.File(path, "r") as hf:
            field_lookup = build_available_field_specs(hf["fields"])

        requested_fields = finalize_requested_field_names(field_lookup, ["density", "velocity_magnitude"])
        self.assertEqual(
            requested_fields,
            ["density", "velocity_magnitude", "density_gradient_magnitude"],
        )

    def test_explicit_q_field_still_inserts_r_and_density_gradient(self) -> None:
        path = self._write_fields_file(include_density=True, include_pressure=True)
        self.addCleanup(lambda: os.unlink(path))

        with h5py.File(path, "r") as hf:
            field_lookup = build_available_field_specs(hf["fields"])

        requested_fields = finalize_requested_field_names(field_lookup, ["q_criterion"])
        self.assertEqual(
            requested_fields,
            ["q_criterion", "r_criterion", "density_gradient_magnitude"],
        )


if __name__ == "__main__":
    unittest.main()
