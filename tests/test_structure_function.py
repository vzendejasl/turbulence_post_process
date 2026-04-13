from __future__ import annotations

import math
import unittest

import numpy as np

from postprocess_fft.spectra import compute_longitudinal_structure_function_from_spectrum


class TestStructureFunctionFromSpectrum(unittest.TestCase):
    def test_uses_physical_wavenumber_for_unit_box(self) -> None:
        k_shells = np.asarray([1.0], dtype=np.float64)
        energy_shells = np.asarray([2.0], dtype=np.float64)
        r_values = np.asarray([0.25], dtype=np.float64)

        _, s_longitudinal = compute_longitudinal_structure_function_from_spectrum(
            k_shells,
            energy_shells,
            r_values,
            domain_length=1.0,
        )

        x = 2.0 * math.pi * 0.25
        expected = 4.0 * 2.0 * (1.0 / 3.0 - (math.sin(x) - x * math.cos(x)) / (x ** 3))
        self.assertAlmostEqual(float(s_longitudinal[0]), expected, places=14)

    def test_returns_zero_at_zero_separation(self) -> None:
        k_shells = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        energy_shells = np.asarray([0.5, 0.25, 0.125], dtype=np.float64)
        r_values = np.asarray([0.0], dtype=np.float64)

        _, s_longitudinal = compute_longitudinal_structure_function_from_spectrum(
            k_shells,
            energy_shells,
            r_values,
            domain_length=1.0,
        )

        self.assertAlmostEqual(float(s_longitudinal[0]), 0.0, places=14)


if __name__ == "__main__":
    unittest.main()
