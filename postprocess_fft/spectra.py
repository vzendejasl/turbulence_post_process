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
