from __future__ import annotations

import os
import tempfile
import unittest

from postprocess_lib.auto_scalars import discover_auto_scalar_inputs
from postprocess_lib.auto_scalars import infer_cycle_identifier


class TestAutoScalarDiscovery(unittest.TestCase):
    def test_infers_cycle_from_parent_directory(self) -> None:
        path = "/tmp/run/cycle_6576/SampledData6576.h5"
        self.assertEqual(infer_cycle_identifier(path), "6576")

    def test_prefers_txt_before_h5_for_matching_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cycle_dir = os.path.join(tmpdir, "cycle_6576")
            os.makedirs(cycle_dir, exist_ok=True)
            velocity_path = os.path.join(cycle_dir, "velocity_sampled_data_uniform_interpolated_cycle_6576.h5")
            density_txt = os.path.join(cycle_dir, "density_sampled_data_uniform_interpolated_cycle_6576.txt")
            density_h5 = os.path.join(cycle_dir, "density_sampled_data_uniform_interpolated_cycle_6576.h5")
            pressure_h5 = os.path.join(cycle_dir, "pressure_sampled_data_uniform_interpolated_cycle_6576.h5")
            for path in (velocity_path, density_txt, density_h5, pressure_h5):
                with open(path, "w", encoding="utf-8"):
                    pass

            discovered = discover_auto_scalar_inputs(velocity_path)
            self.assertEqual(discovered, [density_txt, pressure_h5])

    def test_absent_scalar_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cycle_dir = os.path.join(tmpdir, "cycle_12")
            os.makedirs(cycle_dir, exist_ok=True)
            velocity_path = os.path.join(cycle_dir, "SampledData12.h5")
            with open(velocity_path, "w", encoding="utf-8"):
                pass

            self.assertEqual(discover_auto_scalar_inputs(velocity_path), [])

    def test_explicit_field_skips_auto_discovery_for_that_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cycle_dir = os.path.join(tmpdir, "cycle_9")
            os.makedirs(cycle_dir, exist_ok=True)
            velocity_path = os.path.join(cycle_dir, "SampledData9.h5")
            density_txt = os.path.join(cycle_dir, "density_sampled_data_uniform_interpolated_cycle_9.txt")
            pressure_txt = os.path.join(cycle_dir, "pressure_sampled_data_uniform_interpolated_cycle_9.txt")
            explicit_density = os.path.join(tmpdir, "density_custom_input.txt")
            for path in (velocity_path, density_txt, pressure_txt, explicit_density):
                with open(path, "w", encoding="utf-8"):
                    pass

            discovered = discover_auto_scalar_inputs(velocity_path, explicit_scalar_paths=[explicit_density])
            self.assertEqual(discovered, [pressure_txt])
