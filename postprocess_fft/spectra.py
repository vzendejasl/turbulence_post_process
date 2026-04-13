"""Spectral reductions and diagnostics."""

from __future__ import annotations

import math

import numpy as np
from mpi4py import MPI

from .common import global_mean_energy
from .transform import backward_field
from .transform import forward_field
from .transform import local_integer_wavenumber_mesh


def compute_energy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm):
    nx, _, _ = shape
    norm = float(np.prod(shape))
    energy_density = 0.5 * (
        np.abs(vx_k / norm) ** 2
        + np.abs(vy_k / norm) ** 2
        + np.abs(vz_k / norm) ** 2
    )

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    k_magnitude = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)

    k_max_int = int(math.ceil(nx * 0.5 * math.sqrt(3.0)))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * math.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    local_hist, _ = np.histogram(
        k_magnitude.ravel(order="C"),
        bins=k_bin_edges,
        weights=energy_density.ravel(order="C"),
    )

    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
    return k_bin_centers, global_hist


def compute_enstrophy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    omega_x_k = 1j * (KY_int * vz_kn - KZ_int * vy_kn)
    omega_y_k = 1j * (KZ_int * vx_kn - KX_int * vz_kn)
    omega_z_k = 1j * (KX_int * vy_kn - KY_int * vx_kn)

    enstrophy_density = 0.5 * (
        np.abs(omega_x_k) ** 2 + np.abs(omega_y_k) ** 2 + np.abs(omega_z_k) ** 2
    )
    local_total_enstrophy = np.sum(enstrophy_density, dtype=np.float64)
    total_enstrophy = comm.allreduce(local_total_enstrophy, op=MPI.SUM)

    nx, _, _ = shape
    k_magnitude = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)
    k_max_int = int(math.ceil(nx * 0.5 * math.sqrt(3.0)))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * math.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    local_hist, _ = np.histogram(
        k_magnitude.ravel(order="C"),
        bins=k_bin_edges,
        weights=enstrophy_density.ravel(order="C"),
    )

    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
    return k_bin_centers, global_hist, total_enstrophy


def compute_helicity_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    omega_x_k = 1j * (KY_int * vz_kn - KZ_int * vy_kn)
    omega_y_k = 1j * (KZ_int * vx_kn - KX_int * vz_kn)
    omega_z_k = 1j * (KX_int * vy_kn - KY_int * vx_kn)

    helicity_density = np.real(
        vx_kn * np.conj(omega_x_k) +
        vy_kn * np.conj(omega_y_k) +
        vz_kn * np.conj(omega_z_k)
    )

    nx, _, _ = shape
    k_magnitude = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)
    k_max_int = int(math.ceil(nx * 0.5 * math.sqrt(3.0)))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * math.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    local_hist, _ = np.histogram(
        k_magnitude.ravel(order="C"),
        bins=k_bin_edges,
        weights=helicity_density.ravel(order="C"),
    )

    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
    return k_bin_centers, global_hist


def compute_energy_dissipation_enstrophy(vx_k, vy_k, vz_k, shape, box, comm, root):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    energy_density = 0.5 * (
        np.abs(vx_kn) ** 2 + np.abs(vy_kn) ** 2 + np.abs(vz_kn) ** 2
    )
    local_total_ke = np.sum(energy_density, dtype=np.float64)
    total_ke = comm.allreduce(local_total_ke, op=MPI.SUM)

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    k_squared = KX_int**2 + KY_int**2 + KZ_int**2
    local_diss = np.sum(energy_density * k_squared, dtype=np.float64)
    total_diss = comm.allreduce(local_diss, op=MPI.SUM)

    omega_x_k = 1j * (KY_int * vz_kn - KZ_int * vy_kn)
    omega_y_k = 1j * (KZ_int * vx_kn - KX_int * vz_kn)
    omega_z_k = 1j * (KX_int * vy_kn - KY_int * vx_kn)
    local_enstrophy = 0.5 * np.sum(
        np.abs(omega_x_k) ** 2 + np.abs(omega_y_k) ** 2 + np.abs(omega_z_k) ** 2,
        dtype=np.float64,
    )
    total_enstrophy = comm.allreduce(local_enstrophy, op=MPI.SUM)

    if root:
        print(f"  Total kinetic energy (fourier): {total_ke:.8f}")
        print(f"  Total k^2-weighted energy: {total_diss:.8f}")
        print("  Enstrophy vs total k^2-weighted energy comparison (should be close)")
        print(f"  {total_enstrophy:.8f} {total_diss:.8f}")


def compensate_spectrum(k_centers, values, exponent):
    """Return k^exponent-weighted spectra with the zero mode held at zero."""
    compensated = np.zeros_like(values, dtype=np.float64)
    mask = np.asarray(k_centers) > 0.0
    compensated[mask] = np.asarray(values, dtype=np.float64)[mask] * np.asarray(k_centers, dtype=np.float64)[mask] ** exponent
    return compensated


def compute_longitudinal_structure_function_from_spectrum(
    k_shells,
    energy_shells,
    r_values,
    domain_length,
):
    """Compute isotropic longitudinal S_L(r) from shell-integrated E(k)."""
    k_shells = np.asarray(k_shells, dtype=np.float64)
    energy_shells = np.asarray(energy_shells, dtype=np.float64)
    r_values = np.asarray(r_values, dtype=np.float64)

    if k_shells.shape != energy_shells.shape:
        raise ValueError("k_shells and energy_shells must have the same shape.")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive.")

    k_phys = (2.0 * np.pi / float(domain_length)) * k_shells
    kr = np.outer(r_values, k_phys)

    kernel = np.empty_like(kr, dtype=np.float64)
    small_mask = np.abs(kr) < 1.0e-4
    large_mask = ~small_mask

    if np.any(large_mask):
        kr_large = kr[large_mask]
        kernel[large_mask] = (
            1.0 / 3.0
            - (np.sin(kr_large) - kr_large * np.cos(kr_large)) / (kr_large ** 3)
        )
    if np.any(small_mask):
        kr_small = kr[small_mask]
        kernel[small_mask] = (kr_small ** 2) / 30.0 - (kr_small ** 4) / 840.0

    s_longitudinal = 4.0 * np.sum(kernel * energy_shells[np.newaxis, :], axis=1, dtype=np.float64)
    return r_values, s_longitudinal


def compute_third_order_structure_function_direct(vx, vy, vz, dx, dy, dz):
    """Compute axis-aligned third-order longitudinal structure functions in physical space."""
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    vz = np.asarray(vz, dtype=np.float64)

    if vx.shape != vy.shape or vx.shape != vz.shape:
        raise ValueError("vx, vy, and vz must have the same shape.")

    nx, ny, nz = vx.shape
    if nx != ny or nx != nz:
        raise ValueError("Third-order direct structure function currently requires a cubic grid.")
    if not (np.isclose(dx, dy) and np.isclose(dx, dz)):
        raise ValueError("Third-order direct structure function currently requires uniform spacing dx = dy = dz.")

    max_shift = nx // 2
    shifts = np.arange(max_shift + 1, dtype=np.int64)
    r_values = shifts.astype(np.float64) * float(dx)

    s3_x = np.empty_like(r_values, dtype=np.float64)
    s3_y = np.empty_like(r_values, dtype=np.float64)
    s3_z = np.empty_like(r_values, dtype=np.float64)

    for idx, shift in enumerate(shifts):
        delta_x = np.roll(vx, -int(shift), axis=0) - vx
        delta_y = np.roll(vy, -int(shift), axis=1) - vy
        delta_z = np.roll(vz, -int(shift), axis=2) - vz

        s3_x[idx] = float(np.mean(delta_x ** 3, dtype=np.float64))
        s3_y[idx] = float(np.mean(delta_y ** 3, dtype=np.float64))
        s3_z[idx] = float(np.mean(delta_z ** 3, dtype=np.float64))

    s3_avg = (s3_x + s3_y + s3_z) / 3.0
    return r_values, s3_x, s3_y, s3_z, s3_avg


def _third_order_validate_grid(shape, dx, dy, dz):
    nx, ny, nz = shape
    if nx != ny or nx != nz:
        raise ValueError("Third-order structure function currently requires a cubic grid.")
    if not (np.isclose(dx, dy) and np.isclose(dx, dz)):
        raise ValueError("Third-order structure function currently requires uniform spacing dx = dy = dz.")


def _extract_axis_line_from_local_volume(local_volume, shape, box, axis, comm):
    count = int(shape[axis] // 2) + 1
    local_line = np.zeros(count, dtype=np.float64)
    box_low = tuple(int(value) for value in box.low)
    box_high = tuple(int(value) for value in box.high)

    for shift in range(count):
        global_index = [0, 0, 0]
        global_index[axis] = shift
        if all(box_low[dim] <= global_index[dim] <= box_high[dim] for dim in range(3)):
            local_index = tuple(global_index[dim] - box_low[dim] for dim in range(3))
            local_line[shift] = float(local_volume[local_index])

    return comm.allreduce(local_line, op=MPI.SUM)


def _compute_axis_third_order_fft(component, axis, plan, local_shape, shape, box, comm):
    global_points = float(np.prod(shape))
    component_sq = component ** 2

    component_k = forward_field(plan, component).reshape(local_shape, order="C")
    component_sq_k = forward_field(plan, component_sq).reshape(local_shape, order="C")

    corr_sq_u_local = backward_field(plan, np.conj(component_sq_k) * component_k, local_shape)
    corr_u_sq_local = backward_field(plan, np.conj(component_k) * component_sq_k, local_shape)

    corr_sq_u = _extract_axis_line_from_local_volume(corr_sq_u_local, shape, box, axis, comm) / global_points
    corr_u_sq = _extract_axis_line_from_local_volume(corr_u_sq_local, shape, box, axis, comm) / global_points

    return 3.0 * (corr_sq_u - corr_u_sq)


def compute_third_order_structure_function_fft(plan, local_shape, box, vx, vy, vz, shape, dx, dy, dz, comm):
    """Compute axis-aligned third-order longitudinal structure functions via HeFFTe FFT correlations."""
    _third_order_validate_grid(shape, dx, dy, dz)

    count = int(shape[0] // 2) + 1
    r_values = np.arange(count, dtype=np.float64) * float(dx)

    s3_x = _compute_axis_third_order_fft(vx, 0, plan, local_shape, shape, box, comm)
    s3_y = _compute_axis_third_order_fft(vy, 1, plan, local_shape, shape, box, comm)
    s3_z = _compute_axis_third_order_fft(vz, 2, plan, local_shape, shape, box, comm)
    s3_avg = (s3_x + s3_y + s3_z) / 3.0
    return r_values, s3_x, s3_y, s3_z, s3_avg


def compute_qr_joint_pdf(
    dux_dx,
    dux_dy,
    dux_dz,
    duy_dx,
    duy_dy,
    duy_dz,
    duz_dx,
    duz_dy,
    duz_dz,
    comm,
    bins=256,
):
    """Compute a distributed joint PDF of Frobenius-normalized Q and R invariants."""
    # Use the local Frobenius norm of the velocity-gradient tensor to normalize
    # the invariants at each point, matching the normalized-VGT convention.
    grad_fro_sq = (
        dux_dx**2 + dux_dy**2 + dux_dz**2
        + duy_dx**2 + duy_dy**2 + duy_dz**2
        + duz_dx**2 + duz_dy**2 + duz_dz**2
    )
    local_total_points = int(dux_dx.size)
    global_total_points = comm.allreduce(local_total_points, op=MPI.SUM)
    local_max_grad_fro_sq = float(np.max(grad_fro_sq)) if grad_fro_sq.size > 0 else 0.0
    global_max_grad_fro_sq = comm.allreduce(local_max_grad_fro_sq, op=MPI.MAX)
    filter_fraction = 1.0e-3
    filter_threshold = filter_fraction * global_max_grad_fro_sq

    div_u = dux_dx + duy_dy + duz_dz
    trace_a2 = (
        dux_dx * dux_dx + dux_dy * duy_dx + dux_dz * duz_dx
        + duy_dx * dux_dy + duy_dy * duy_dy + duy_dz * duz_dy
        + duz_dx * dux_dz + duz_dy * duy_dz + duz_dz * duz_dz
    )
    q_local = 0.5 * (div_u**2 - trace_a2)
    r_local = -(
        dux_dx * (duy_dy * duz_dz - duy_dz * duz_dy)
        - dux_dy * (duy_dx * duz_dz - duy_dz * duz_dx)
        + dux_dz * (duy_dx * duz_dy - duy_dy * duz_dx)
    )

    if global_max_grad_fro_sq > 1.0e-30:
        retain_mask = grad_fro_sq >= filter_threshold
    else:
        retain_mask = np.ones_like(grad_fro_sq, dtype=bool)

    grad_fro_sq_retained = np.maximum(grad_fro_sq[retain_mask], 1.0e-30)
    q_norm_local = q_local[retain_mask] / grad_fro_sq_retained
    r_norm_local = r_local[retain_mask] / (grad_fro_sq_retained ** 1.5)

    local_retained_samples = int(q_norm_local.size)
    global_retained_samples = comm.allreduce(local_retained_samples, op=MPI.SUM)

    if global_retained_samples == 0:
        if comm.Get_rank() != 0:
            return None
        zero_edges = np.linspace(-1.0, 1.0, int(bins) + 1, dtype=np.float64)
        zero_centers = 0.5 * (zero_edges[:-1] + zero_edges[1:])
        zero_hist = np.zeros((int(bins), int(bins)), dtype=np.float64)
        return {
            "q_edges": zero_edges,
            "r_edges": zero_edges,
            "q_centers": zero_centers,
            "r_centers": zero_centers,
            "counts": zero_hist,
            "joint_pdf": zero_hist,
            "total_samples": int(global_retained_samples),
            "total_points": int(global_total_points),
            "retained_samples": int(global_retained_samples),
            "retained_fraction": 0.0,
            "max_grad_fro_sq": float(global_max_grad_fro_sq),
            "filter_fraction": float(filter_fraction),
            "filter_threshold": float(filter_threshold),
            "q_min": -1.0,
            "q_max": 1.0,
            "r_min": -1.0,
            "r_max": 1.0,
        }

    q_local_min = float(np.min(q_norm_local)) if q_norm_local.size > 0 else 0.0
    q_local_max = float(np.max(q_norm_local)) if q_norm_local.size > 0 else 0.0
    r_local_min = float(np.min(r_norm_local)) if r_norm_local.size > 0 else 0.0
    r_local_max = float(np.max(r_norm_local)) if r_norm_local.size > 0 else 0.0

    q_global_min = comm.allreduce(q_local_min, op=MPI.MIN)
    q_global_max = comm.allreduce(q_local_max, op=MPI.MAX)
    r_global_min = comm.allreduce(r_local_min, op=MPI.MIN)
    r_global_max = comm.allreduce(r_local_max, op=MPI.MAX)

    if np.isclose(q_global_min, q_global_max):
        delta = 1.0 if np.isclose(q_global_min, 0.0) else 1.0e-6 * abs(q_global_min)
        q_global_min -= delta
        q_global_max += delta
    if np.isclose(r_global_min, r_global_max):
        delta = 1.0 if np.isclose(r_global_min, 0.0) else 1.0e-6 * abs(r_global_min)
        r_global_min -= delta
        r_global_max += delta

    q_edges = np.linspace(q_global_min, q_global_max, int(bins) + 1, dtype=np.float64)
    r_edges = np.linspace(r_global_min, r_global_max, int(bins) + 1, dtype=np.float64)

    local_hist, _, _ = np.histogram2d(
        q_norm_local.ravel(order="C"),
        r_norm_local.ravel(order="C"),
        bins=(q_edges, r_edges),
    )
    local_hist = np.ascontiguousarray(local_hist, dtype=np.float64)
    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)

    if comm.Get_rank() != 0:
        return None

    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    bin_area = np.outer(np.diff(q_edges), np.diff(r_edges))
    joint_pdf = global_hist / (float(global_retained_samples) * bin_area)

    return {
        "q_edges": q_edges,
        "r_edges": r_edges,
        "q_centers": q_centers,
        "r_centers": r_centers,
        "counts": global_hist,
        "joint_pdf": joint_pdf,
        "total_samples": int(global_retained_samples),
        "total_points": int(global_total_points),
        "retained_samples": int(global_retained_samples),
        "retained_fraction": float(global_retained_samples / float(global_total_points)) if global_total_points > 0 else 0.0,
        "max_grad_fro_sq": float(global_max_grad_fro_sq),
        "filter_fraction": float(filter_fraction),
        "filter_threshold": float(filter_threshold),
        "q_min": float(q_global_min),
        "q_max": float(q_global_max),
        "r_min": float(r_global_min),
        "r_max": float(r_global_max),
    }
