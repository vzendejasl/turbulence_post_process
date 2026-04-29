from __future__ import annotations

import unittest

import numpy as np

from tools.replot_slice_data import _apply_normalization


class TestReplotNormalization(unittest.TestCase):
    def test_global_rms_applies_to_raw_saved_values(self) -> None:
        saved = {
            "values": np.array([[2.0, 4.0]], dtype=np.float64),
            "attrs": {
                "field_name": "velocity_magnitude",
                "field_family": "velocity",
                "base_plot_label": r"$|\mathbf{u}|$",
                "value_normalization": "none",
                "global_rms": 2.0,
                "global_min": 0.0,
                "global_max": 4.0,
            },
        }

        normalized = _apply_normalization(saved, "global_rms", False, print_stats=False)

        np.testing.assert_allclose(normalized["values"], np.array([[1.0, 2.0]], dtype=np.float64))
        self.assertEqual(normalized["attrs"]["value_normalization"], "global_rms")
        self.assertEqual(normalized["attrs"]["display_normalization"], "global_rms")
        self.assertAlmostEqual(float(normalized["attrs"]["global_max"]), 2.0)

    def test_none_restores_raw_values_from_legacy_rms_saved_slice(self) -> None:
        saved = {
            "values": np.array([[1.0, 2.0]], dtype=np.float64),
            "attrs": {
                "field_name": "velocity_magnitude",
                "field_family": "velocity",
                "base_plot_label": r"$|\mathbf{u}|$",
                "value_normalization": "global_rms",
                "global_rms": 2.0,
                "global_min": 0.0,
                "global_max": 2.0,
            },
        }

        restored = _apply_normalization(saved, "none", False, print_stats=False)

        np.testing.assert_allclose(restored["values"], np.array([[2.0, 4.0]], dtype=np.float64))
        self.assertEqual(restored["attrs"]["value_normalization"], "none")
        self.assertEqual(restored["attrs"]["display_normalization"], "none")
        self.assertAlmostEqual(float(restored["attrs"]["global_max"]), 4.0)

    def test_global_std_applies_mean_subtracted_std_scaling(self) -> None:
        saved = {
            "values": np.array([[2.0, 4.0]], dtype=np.float64),
            "attrs": {
                "field_name": "velocity_magnitude",
                "field_family": "velocity",
                "base_plot_label": r"$|\mathbf{u}|$",
                "value_normalization": "none",
                "global_mean": 1.0,
                "global_std": 2.0,
                "global_min": 0.0,
                "global_max": 4.0,
            },
        }

        normalized = _apply_normalization(saved, "global_std", False, print_stats=False)

        np.testing.assert_allclose(normalized["values"], np.array([[0.5, 1.5]], dtype=np.float64))
        self.assertEqual(normalized["attrs"]["value_normalization"], "global_std")
        self.assertEqual(normalized["attrs"]["display_normalization"], "global_std")
        self.assertAlmostEqual(float(normalized["attrs"]["global_min"]), -0.5)
        self.assertAlmostEqual(float(normalized["attrs"]["global_max"]), 1.5)

    def test_none_restores_raw_values_from_saved_std_slice(self) -> None:
        saved = {
            "values": np.array([[0.5, 1.5]], dtype=np.float64),
            "attrs": {
                "field_name": "velocity_magnitude",
                "field_family": "velocity",
                "base_plot_label": r"$|\mathbf{u}|$",
                "value_normalization": "global_std",
                "global_mean": 1.0,
                "global_std": 2.0,
                "global_min": -0.5,
                "global_max": 1.5,
            },
        }

        restored = _apply_normalization(saved, "none", False, print_stats=False)

        np.testing.assert_allclose(restored["values"], np.array([[2.0, 4.0]], dtype=np.float64))
        self.assertEqual(restored["attrs"]["value_normalization"], "none")
        self.assertEqual(restored["attrs"]["display_normalization"], "none")
        self.assertAlmostEqual(float(restored["attrs"]["global_min"]), 0.0)
        self.assertAlmostEqual(float(restored["attrs"]["global_max"]), 4.0)


if __name__ == "__main__":
    unittest.main()
