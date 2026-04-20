from __future__ import annotations

import os
import tempfile
import unittest

import h5py
import numpy as np

from postprocess_vis.field_specs import build_available_field_specs
from postprocess_vis.field_specs import default_requested_field_names


class TestSliceFieldSpecs(unittest.TestCase):
    def _write_fields_file(self, include_density: bool) -> str:
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
        return handle.name

    def test_density_adds_density_gradient_field(self) -> None:
        path = self._write_fields_file(include_density=True)
        self.addCleanup(lambda: os.unlink(path))

        with h5py.File(path, "r") as hf:
            field_lookup = build_available_field_specs(hf["fields"])

        self.assertIn("density", field_lookup)
        self.assertIn("density_gradient_magnitude", field_lookup)
        self.assertEqual(field_lookup["density_gradient_magnitude"][3], "density_gradient")

        requested_fields = default_requested_field_names(field_lookup)
        self.assertEqual(
            requested_fields[:5],
            [
                "velocity_magnitude",
                "vorticity_magnitude",
                "q_criterion",
                "r_criterion",
                "density_gradient_magnitude",
            ],
        )
        self.assertIn("density", requested_fields)

    def test_density_gradient_not_registered_without_density(self) -> None:
        path = self._write_fields_file(include_density=False)
        self.addCleanup(lambda: os.unlink(path))

        with h5py.File(path, "r") as hf:
            field_lookup = build_available_field_specs(hf["fields"])

        self.assertNotIn("density_gradient_magnitude", field_lookup)
        self.assertNotIn("density_gradient_magnitude", default_requested_field_names(field_lookup))


if __name__ == "__main__":
    unittest.main()
