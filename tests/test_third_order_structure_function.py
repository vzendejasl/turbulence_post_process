from __future__ import annotations

import unittest

import numpy as np
from mpi4py import MPI

from postprocess_fft.spectra import compute_shell_averaged_third_order_structure_function_fft
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

    def test_shell_average_matches_axis_average_on_the_first_shell(self) -> None:
        rank = MPI.COMM_WORLD.rank
        n = 8
        if rank == 0:
            grid = np.arange(n, dtype=np.float64)
            xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
            vx = np.sin(2.0 * np.pi * xx / 8.0) + 0.2 * np.cos(2.0 * np.pi * yy / 8.0)
            vy = np.cos(2.0 * np.pi * yy / 8.0)
            vz = np.sin(2.0 * np.pi * zz / 8.0)
            max_shift = n // 2
            expected_shell = {}
            expected_counts = {}
            for mx in range(-max_shift, max_shift + 1):
                for my in range(-max_shift, max_shift + 1):
                    for mz in range(-max_shift, max_shift + 1):
                        if mx == 0 and my == 0 and mz == 0:
                            continue
                        if not ((mx > 0) or ((mx == 0) and (my > 0)) or ((mx == 0) and (my == 0) and (mz > 0))):
                            continue
                        radius = float(np.sqrt(mx * mx + my * my + mz * mz))
                        shell_index = int(np.rint(radius))
                        if shell_index < 1 or shell_index > max_shift:
                            continue
                        direction = np.array([mx, my, mz], dtype=np.float64) / radius
                        delta_u_l = (
                            direction[0] * (np.roll(vx, -mx, axis=0) - vx)
                            + direction[1] * (np.roll(vy, -my, axis=1) - vy)
                            + direction[2] * (np.roll(vz, -mz, axis=2) - vz)
                        )
                        expected_shell.setdefault(shell_index, []).append(float(np.mean(delta_u_l ** 3, dtype=np.float64)))
                        expected_counts[shell_index] = expected_counts.get(shell_index, 0) + 1
            expected_r_values = np.array(sorted(expected_shell.keys()), dtype=np.float64) / float(n)
            expected_shell_values = np.array(
                [np.mean(expected_shell[index], dtype=np.float64) for index in sorted(expected_shell.keys())],
                dtype=np.float64,
            )
            expected_shell_counts = np.array(
                [expected_counts[index] for index in sorted(expected_shell.keys())],
                dtype=np.float64,
            )
        else:
            vx = vy = vz = None
            expected_r_values = expected_shell_values = expected_shell_counts = None

        comm = MPI.COMM_WORLD
        shape = (n, n, n)
        proc_grid = choose_proc_grid(shape, comm.size)
        boxes = build_boxes(shape, proc_grid)
        local_box = boxes[comm.rank]
        local_shape = box_shape(local_box)
        local_vx = scatter_field(vx if comm.rank == 0 else None, boxes, comm).reshape(local_shape, order="C")
        local_vy = scatter_field(vy if comm.rank == 0 else None, boxes, comm).reshape(local_shape, order="C")
        local_vz = scatter_field(vz if comm.rank == 0 else None, boxes, comm).reshape(local_shape, order="C")
        try:
            backend = get_backend("heffte_fftw")
        except RuntimeError:
            backend = heffte.backend.stock
        plan = heffte.fft3d(backend, local_box, local_box, comm)

        shell_result = (
            compute_shell_averaged_third_order_structure_function_fft(
                plan,
                local_shape,
                local_box,
                local_vx,
                local_vy,
                local_vz,
                shape,
                1.0 / 8.0,
                1.0 / 8.0,
                1.0 / 8.0,
                comm,
            )
        )
        shell_r_values, s3_shell, shell_counts = MPI.COMM_WORLD.bcast(shell_result, root=0)
        expected_r_values = MPI.COMM_WORLD.bcast(expected_r_values, root=0)
        expected_shell_values = MPI.COMM_WORLD.bcast(expected_shell_values, root=0)
        expected_shell_counts = MPI.COMM_WORLD.bcast(expected_shell_counts, root=0)

        np.testing.assert_allclose(shell_r_values, expected_r_values, atol=1.0e-15, rtol=1.0e-15)
        np.testing.assert_allclose(s3_shell, expected_shell_values, atol=1.0e-12, rtol=1.0e-12)
        self.assertEqual(shell_counts.shape, expected_shell_counts.shape)
        self.assertTrue(np.all(shell_counts > 0.0))


if __name__ == "__main__":
    unittest.main()
