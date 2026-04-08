"""Spectral reductions and diagnostics."""

from __future__ import annotations

import math

import numpy as np
from mpi4py import MPI

from .common import global_mean_energy
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
    """Compute a distributed joint PDF of normalized Q and R invariants."""
    s12 = 0.5 * (dux_dy + duy_dx)
    s13 = 0.5 * (dux_dz + duz_dx)
    s23 = 0.5 * (duy_dz + duz_dy)
    sij_sij = dux_dx**2 + duy_dy**2 + duz_dz**2 + 2.0 * (s12**2 + s13**2 + s23**2)

    local_count = int(dux_dx.size)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    local_sum_sij_sij = float(np.sum(sij_sij, dtype=np.float64))
    global_sum_sij_sij = comm.allreduce(local_sum_sij_sij, op=MPI.SUM)
    avg_sij_sij = global_sum_sij_sij / float(global_count) if global_count > 0 else 0.0
    sij_sij_scale = max(abs(avg_sij_sij), 1.0e-30)

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

    q_norm_local = q_local / sij_sij_scale
    r_norm_local = r_local / (sij_sij_scale ** 1.5)

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
    joint_pdf = global_hist / (float(global_count) * bin_area)

    return {
        "q_edges": q_edges,
        "r_edges": r_edges,
        "q_centers": q_centers,
        "r_centers": r_centers,
        "counts": global_hist,
        "joint_pdf": joint_pdf,
        "avg_sij_sij": float(avg_sij_sij),
        "total_samples": int(global_count),
        "q_min": float(q_global_min),
        "q_max": float(q_global_max),
        "r_min": float(r_global_min),
        "r_max": float(r_global_max),
    }
