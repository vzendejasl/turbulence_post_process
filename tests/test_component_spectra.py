from __future__ import annotations

import unittest

import numpy as np

try:
    from mpi4py import MPI
    from postprocess_fft.layout import build_boxes
    from postprocess_fft.spectra import compute_energy_component_spectra_from_modes
    from postprocess_fft.spectra import compute_energy_spectrum_from_modes
    from postprocess_fft.spectra import compute_enstrophy_component_spectra_from_modes
    from postprocess_fft.spectra import compute_enstrophy_spectrum_from_modes

    _MPI_SPECTRA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    MPI = None
    build_boxes = None
    compute_energy_component_spectra_from_modes = None
    compute_energy_spectrum_from_modes = None
    compute_enstrophy_component_spectra_from_modes = None
    compute_enstrophy_spectrum_from_modes = None
    _MPI_SPECTRA_IMPORT_ERROR = exc


@unittest.skipIf(_MPI_SPECTRA_IMPORT_ERROR is not None, f"MPI spectra stack unavailable: {_MPI_SPECTRA_IMPORT_ERROR}")
class TestComponentSpectra(unittest.TestCase):
    def test_energy_component_spectra_sum_to_total_spectrum(self) -> None:
        shape = (4, 4, 4)
        box = build_boxes(shape, (1, 1, 1))[0]
        comm = MPI.COMM_SELF

        vx_k = np.arange(64, dtype=np.float64).reshape(shape).astype(np.complex128)
        vy_k = (2.0 * vx_k + 1.0j).astype(np.complex128)
        vz_k = (-0.5 * vx_k + 2.0j).astype(np.complex128)

        k_total, e_total = compute_energy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm)
        k_comp, e_x, e_y, e_z = compute_energy_component_spectra_from_modes(vx_k, vy_k, vz_k, shape, box, comm)

        self.assertTrue(np.allclose(k_total, k_comp))
        self.assertTrue(np.allclose(e_total, e_x + e_y + e_z))

    def test_enstrophy_component_spectra_sum_to_total_spectrum(self) -> None:
        shape = (4, 4, 4)
        box = build_boxes(shape, (1, 1, 1))[0]
        comm = MPI.COMM_SELF

        grid = np.arange(64, dtype=np.float64).reshape(shape)
        vx_k = (grid + 1.0j * (grid + 1.0)).astype(np.complex128)
        vy_k = (0.5 * grid - 2.0j).astype(np.complex128)
        vz_k = (-1.5 * grid + 0.25j * grid).astype(np.complex128)

        k_total, enst_total, total_enstrophy = compute_enstrophy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm)
        k_comp, enst_x, enst_y, enst_z, component_total_enstrophy = compute_enstrophy_component_spectra_from_modes(
            vx_k,
            vy_k,
            vz_k,
            shape,
            box,
            comm,
        )

        self.assertTrue(np.allclose(k_total, k_comp))
        self.assertTrue(np.allclose(enst_total, enst_x + enst_y + enst_z))
        self.assertAlmostEqual(float(total_enstrophy), float(component_total_enstrophy), places=14)


if __name__ == "__main__":
    unittest.main()
