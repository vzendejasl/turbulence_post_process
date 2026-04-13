from __future__ import annotations

import unittest

import numpy as np
from mpi4py import MPI

from postprocess_fft.spectra import compute_third_order_structure_function_direct
from postprocess_fft.spectra import compute_third_order_structure_function_fft
from postprocess_fft.common import heffte
from postprocess_fft.layout import box_shape
from postprocess_fft.layout import build_boxes
from postprocess_fft.layout import choose_proc_grid
from postprocess_fft.layout import scatter_field
from postprocess_fft.transform import get_backend


class TestThirdOrderStructureFunctionDirect(unittest.TestCase):
    def test_constant_field_is_zero(self) -> None:
        vx = np.ones((8, 8, 8), dtype=np.float64)
        vy = 2.0 * np.ones((8, 8, 8), dtype=np.float64)
        vz = -3.0 * np.ones((8, 8, 8), dtype=np.float64)

        r_values, s3_x, s3_y, s3_z, s3_avg = compute_third_order_structure_function_direct(
            vx, vy, vz, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0
        )

        self.assertEqual(len(r_values), 5)
        np.testing.assert_allclose(s3_x, 0.0, atol=1.0e-15)
        np.testing.assert_allclose(s3_y, 0.0, atol=1.0e-15)
        np.testing.assert_allclose(s3_z, 0.0, atol=1.0e-15)
        np.testing.assert_allclose(s3_avg, 0.0, atol=1.0e-15)

    def test_zero_separation_is_zero(self) -> None:
        grid = np.arange(8, dtype=np.float64)
        xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
        vx = np.sin(2.0 * np.pi * xx / 8.0)
        vy = np.cos(2.0 * np.pi * yy / 8.0)
        vz = np.sin(2.0 * np.pi * zz / 8.0)

        r_values, s3_x, s3_y, s3_z, s3_avg = compute_third_order_structure_function_direct(
            vx, vy, vz, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0
        )

        self.assertAlmostEqual(float(r_values[0]), 0.0, places=14)
        self.assertAlmostEqual(float(s3_x[0]), 0.0, places=14)
        self.assertAlmostEqual(float(s3_y[0]), 0.0, places=14)
        self.assertAlmostEqual(float(s3_z[0]), 0.0, places=14)
        self.assertAlmostEqual(float(s3_avg[0]), 0.0, places=14)

    def test_velocity_sign_flip_flips_structure_function_sign(self) -> None:
        grid = np.arange(8, dtype=np.float64)
        xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
        vx = np.sin(2.0 * np.pi * xx / 8.0)
        vy = np.cos(2.0 * np.pi * yy / 8.0)
        vz = np.sin(4.0 * np.pi * zz / 8.0)

        _, s3_x, s3_y, s3_z, s3_avg = compute_third_order_structure_function_direct(
            vx, vy, vz, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0
        )
        _, neg_s3_x, neg_s3_y, neg_s3_z, neg_s3_avg = compute_third_order_structure_function_direct(
            -vx, -vy, -vz, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0
        )

        np.testing.assert_allclose(neg_s3_x, -s3_x, atol=1.0e-14)
        np.testing.assert_allclose(neg_s3_y, -s3_y, atol=1.0e-14)
        np.testing.assert_allclose(neg_s3_z, -s3_z, atol=1.0e-14)
        np.testing.assert_allclose(neg_s3_avg, -s3_avg, atol=1.0e-14)

    def test_directional_component_field_only_populates_one_direction(self) -> None:
        grid = np.arange(8, dtype=np.float64)
        xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
        vx = np.sin(2.0 * np.pi * xx / 8.0) + 0.25 * np.sin(4.0 * np.pi * xx / 8.0)
        vy = np.zeros_like(vx)
        vz = np.zeros_like(vx)

        _, s3_x, s3_y, s3_z, _ = compute_third_order_structure_function_direct(
            vx, vy, vz, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0
        )

        self.assertTrue(np.any(np.abs(s3_x) > 1.0e-12))
        np.testing.assert_allclose(s3_y, 0.0, atol=1.0e-15)
        np.testing.assert_allclose(s3_z, 0.0, atol=1.0e-15)


class TestThirdOrderStructureFunctionFFT(unittest.TestCase):
    def _run_fft_helper(self, vx_global, vy_global, vz_global):
        comm = MPI.COMM_WORLD
        rank = comm.rank
        shape = vx_global.shape if rank == 0 else None
        shape = comm.bcast(shape, root=0)
        proc_grid = choose_proc_grid(shape, comm.size)
        boxes = build_boxes(shape, proc_grid)
        local_box = boxes[rank]
        local_shape = box_shape(local_box)

        local_vx = scatter_field(vx_global if rank == 0 else None, boxes, comm).reshape(local_shape, order="C")
        local_vy = scatter_field(vy_global if rank == 0 else None, boxes, comm).reshape(local_shape, order="C")
        local_vz = scatter_field(vz_global if rank == 0 else None, boxes, comm).reshape(local_shape, order="C")

        try:
            backend = get_backend("heffte_fftw")
        except RuntimeError:
            backend = heffte.backend.stock
        plan = heffte.fft3d(backend, local_box, local_box, comm)
        dx = dy = dz = 1.0 / float(shape[0])

        return compute_third_order_structure_function_fft(
            plan,
            local_shape,
            local_box,
            local_vx,
            local_vy,
            local_vz,
            shape,
            dx,
            dy,
            dz,
            comm,
        )

    def test_fft_matches_direct_on_synthetic_velocity_field(self) -> None:
        rank = MPI.COMM_WORLD.rank
        if rank == 0:
            grid = np.arange(8, dtype=np.float64)
            xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
            vx = np.sin(2.0 * np.pi * xx / 8.0) + 0.2 * np.cos(4.0 * np.pi * yy / 8.0)
            vy = np.cos(2.0 * np.pi * yy / 8.0) + 0.1 * np.sin(2.0 * np.pi * zz / 8.0)
            vz = np.sin(2.0 * np.pi * zz / 8.0) + 0.15 * np.cos(2.0 * np.pi * xx / 8.0)
            direct = compute_third_order_structure_function_direct(vx, vy, vz, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0)
        else:
            vx = vy = vz = None
            direct = None

        direct = MPI.COMM_WORLD.bcast(direct, root=0)
        fft_result = self._run_fft_helper(vx, vy, vz)

        for fft_values, direct_values in zip(fft_result, direct):
            np.testing.assert_allclose(fft_values, direct_values, atol=1.0e-12, rtol=1.0e-12)

    def test_periodic_linear_field_matches_analytic_third_order_formula(self) -> None:
        rank = MPI.COMM_WORLD.rank
        n = 8
        if rank == 0:
            grid = np.arange(n, dtype=np.float64) / float(n)
            xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
            vx = xx.copy()
            vy = yy.copy()
            vz = zz.copy()
        else:
            vx = vy = vz = None

        r_values, s3_x, s3_y, s3_z, s3_avg = self._run_fft_helper(vx, vy, vz)
        expected = -r_values + 3.0 * (r_values ** 2) - 2.0 * (r_values ** 3)

        np.testing.assert_allclose(s3_x, expected, atol=1.0e-12, rtol=1.0e-12)
        np.testing.assert_allclose(s3_y, expected, atol=1.0e-12, rtol=1.0e-12)
        np.testing.assert_allclose(s3_z, expected, atol=1.0e-12, rtol=1.0e-12)
        np.testing.assert_allclose(s3_avg, expected, atol=1.0e-12, rtol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
