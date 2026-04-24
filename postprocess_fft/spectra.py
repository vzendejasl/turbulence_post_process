"""Spectral reductions and diagnostics."""

from __future__ import annotations

import math

import numpy as np
from mpi4py import MPI

from .common import global_mean_energy
from .transform import backward_field
from .transform import forward_field
from .transform import local_integer_wavenumber_mesh
from .transform import local_wavenumber_mesh


def _shell_bin_geometry(shape, box):
    nx, _, _ = shape
    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, box)
    k_magnitude = np.sqrt(KX_int**2 + KY_int**2 + KZ_int**2)
    k_max_int = int(math.ceil(nx * 0.5 * math.sqrt(3.0)))
    k_bin_edges = np.linspace(0.5, k_max_int + 0.5, k_max_int + 1)
    if nx * 0.5 * math.sqrt(3.0) < k_bin_edges[-2]:
        k_bin_edges = k_bin_edges[:-1]
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
    return k_magnitude, k_bin_edges, k_bin_centers


def _reduce_shell_histogram(k_magnitude, k_bin_edges, weights, comm):
    local_hist, _ = np.histogram(
        k_magnitude.ravel(order="C"),
        bins=k_bin_edges,
        weights=np.asarray(weights, dtype=np.float64).ravel(order="C"),
    )

    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
    return global_hist


def _component_shell_spectra(component_x_density, component_y_density, component_z_density, shape, box, comm):
    k_magnitude, k_bin_edges, k_bin_centers = _shell_bin_geometry(shape, box)
    global_hist_x = _reduce_shell_histogram(k_magnitude, k_bin_edges, component_x_density, comm)
    global_hist_y = _reduce_shell_histogram(k_magnitude, k_bin_edges, component_y_density, comm)
    global_hist_z = _reduce_shell_histogram(k_magnitude, k_bin_edges, component_z_density, comm)
    return k_bin_centers, global_hist_x, global_hist_y, global_hist_z


def compute_energy_component_spectra_from_modes(vx_k, vy_k, vz_k, shape, box, comm):
    norm = float(np.prod(shape))
    component_x_density = 0.5 * np.abs(vx_k / norm) ** 2
    component_y_density = 0.5 * np.abs(vy_k / norm) ** 2
    component_z_density = 0.5 * np.abs(vz_k / norm) ** 2
    return _component_shell_spectra(component_x_density, component_y_density, component_z_density, shape, box, comm)


def compute_energy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, comm):
    k_bin_centers, global_hist_x, global_hist_y, global_hist_z = compute_energy_component_spectra_from_modes(
        vx_k,
        vy_k,
        vz_k,
        shape,
        box,
        comm,
    )
    global_hist = None if global_hist_x is None else global_hist_x + global_hist_y + global_hist_z
    return k_bin_centers, global_hist

def compute_enstrophy_component_spectra_from_modes(vx_k, vy_k, vz_k, shape, box, dx, dy, dz, comm):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    KX_phys, KY_phys, KZ_phys = local_wavenumber_mesh(shape, box, dx, dy, dz)
    omega_x_k = 1j * (KY_phys * vz_kn - KZ_phys * vy_kn)
    omega_y_k = 1j * (KZ_phys * vx_kn - KX_phys * vz_kn)
    omega_z_k = 1j * (KX_phys * vy_kn - KY_phys * vx_kn)

    component_x_density = 0.5 * np.abs(omega_x_k) ** 2
    component_y_density = 0.5 * np.abs(omega_y_k) ** 2
    component_z_density = 0.5 * np.abs(omega_z_k) ** 2
    local_total_enstrophy = np.sum(
        component_x_density + component_y_density + component_z_density,
        dtype=np.float64,
    )
    total_enstrophy = comm.allreduce(local_total_enstrophy, op=MPI.SUM)
    k_bin_centers, global_hist_x, global_hist_y, global_hist_z = _component_shell_spectra(
        component_x_density,
        component_y_density,
        component_z_density,
        shape,
        box,
        comm,
    )
    return k_bin_centers, global_hist_x, global_hist_y, global_hist_z, total_enstrophy


def compute_enstrophy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, dx, dy, dz, comm):
    k_bin_centers, global_hist_x, global_hist_y, global_hist_z, total_enstrophy = (
        compute_enstrophy_component_spectra_from_modes(vx_k, vy_k, vz_k, shape, box, dx, dy, dz, comm)
    )
    global_hist = None if global_hist_x is None else global_hist_x + global_hist_y + global_hist_z
    return k_bin_centers, global_hist, total_enstrophy


def compute_helicity_spectrum_from_modes(vx_k, vy_k, vz_k, shape, box, dx, dy, dz, comm):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    KX_phys, KY_phys, KZ_phys = local_wavenumber_mesh(shape, box, dx, dy, dz)
    omega_x_k = 1j * (KY_phys * vz_kn - KZ_phys * vy_kn)
    omega_y_k = 1j * (KZ_phys * vx_kn - KX_phys * vz_kn)
    omega_z_k = 1j * (KX_phys * vy_kn - KY_phys * vx_kn)

    helicity_density = np.real(
        vx_kn * np.conj(omega_x_k) +
        vy_kn * np.conj(omega_y_k) +
        vz_kn * np.conj(omega_z_k)
    )

    k_magnitude, k_bin_edges, k_bin_centers = _shell_bin_geometry(shape, box)

    local_hist, _ = np.histogram(
        k_magnitude.ravel(order="C"),
        bins=k_bin_edges,
        weights=helicity_density.ravel(order="C"),
    )

    global_hist = np.zeros_like(local_hist) if comm.Get_rank() == 0 else None
    comm.Reduce(local_hist, global_hist, op=MPI.SUM, root=0)
    return k_bin_centers, global_hist


def compute_energy_dissipation_enstrophy(vx_k, vy_k, vz_k, shape, box, dx, dy, dz, comm, root):
    norm = float(np.prod(shape))
    vx_kn = vx_k / norm
    vy_kn = vy_k / norm
    vz_kn = vz_k / norm

    energy_density = 0.5 * (
        np.abs(vx_kn) ** 2 + np.abs(vy_kn) ** 2 + np.abs(vz_kn) ** 2
    )
    local_total_ke = np.sum(energy_density, dtype=np.float64)
    total_ke = comm.allreduce(local_total_ke, op=MPI.SUM)

    KX_phys, KY_phys, KZ_phys = local_wavenumber_mesh(shape, box, dx, dy, dz)
    k_squared = KX_phys**2 + KY_phys**2 + KZ_phys**2
    local_diss = np.sum(energy_density * k_squared, dtype=np.float64)
    total_diss = comm.allreduce(local_diss, op=MPI.SUM)

    omega_x_k = 1j * (KY_phys * vz_kn - KZ_phys * vy_kn)
    omega_y_k = 1j * (KZ_phys * vx_kn - KX_phys * vz_kn)
    omega_z_k = 1j * (KX_phys * vy_kn - KY_phys * vx_kn)
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


def _axis_structure_function_shifts(point_count, full_domain):
    """Return axis-aligned periodic shifts for either half-box or full-box sampling."""
    max_shift = int(point_count) if full_domain else int(point_count // 2)
    return np.arange(max_shift + 1, dtype=np.int64)


def compute_third_order_structure_function_direct(vx, vy, vz, dx, dy, dz, full_domain=False):
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

    shifts = _axis_structure_function_shifts(nx, full_domain)
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


def _extract_axis_line_from_local_volume(local_volume, shape, box, axis, comm, full_domain=False):
    shifts = _axis_structure_function_shifts(shape[axis], full_domain)
    count = int(len(shifts))
    local_line = np.zeros(count, dtype=np.float64)
    box_low = tuple(int(value) for value in box.low)
    box_high = tuple(int(value) for value in box.high)

    axis_count = int(shape[axis])
    for output_index, shift in enumerate(shifts):
        global_index = [0, 0, 0]
        global_index[axis] = int(shift % axis_count)
        if all(box_low[dim] <= global_index[dim] <= box_high[dim] for dim in range(3)):
            local_index = tuple(global_index[dim] - box_low[dim] for dim in range(3))
            local_line[output_index] = float(local_volume[local_index])

    return comm.allreduce(local_line, op=MPI.SUM)


def _compute_axis_third_order_fft(component, axis, plan, local_shape, shape, box, comm, full_domain=False):
    global_points = float(np.prod(shape))
    component_sq = component ** 2

    component_k = forward_field(plan, component).reshape(local_shape, order="C")
    component_sq_k = forward_field(plan, component_sq).reshape(local_shape, order="C")

    corr_sq_u_local = backward_field(plan, np.conj(component_sq_k) * component_k, local_shape)
    corr_u_sq_local = backward_field(plan, np.conj(component_k) * component_sq_k, local_shape)

    corr_sq_u = _extract_axis_line_from_local_volume(
        corr_sq_u_local,
        shape,
        box,
        axis,
        comm,
        full_domain=full_domain,
    ) / global_points
    corr_u_sq = _extract_axis_line_from_local_volume(
        corr_u_sq_local,
        shape,
        box,
        axis,
        comm,
        full_domain=full_domain,
    ) / global_points

    return 3.0 * (corr_sq_u - corr_u_sq)


def _local_shortest_periodic_displacements(shape, box):
    """Return shortest periodic displacement components for one local correlation volume."""
    local_shape = (
        int(box.high[0] - box.low[0] + 1),
        int(box.high[1] - box.low[1] + 1),
        int(box.high[2] - box.low[2] + 1),
    )
    axes = []
    for axis in range(3):
        start = int(box.low[axis])
        stop = int(box.high[axis]) + 1
        coords = np.arange(start, stop, dtype=np.int64)
        half = int(shape[axis] // 2)
        coords = np.where(coords <= half, coords, coords - int(shape[axis]))
        axes.append(coords)

    mx = np.broadcast_to(axes[0][:, np.newaxis, np.newaxis], local_shape)
    my = np.broadcast_to(axes[1][np.newaxis, :, np.newaxis], local_shape)
    mz = np.broadcast_to(axes[2][np.newaxis, np.newaxis, :], local_shape)
    return mx, my, mz


def compute_third_order_structure_function_fft(
    plan,
    local_shape,
    box,
    vx,
    vy,
    vz,
    shape,
    dx,
    dy,
    dz,
    comm,
    full_domain=False,
):
    """Compute axis-aligned third-order longitudinal structure functions via HeFFTe FFT correlations."""
    _third_order_validate_grid(shape, dx, dy, dz)

    shifts = _axis_structure_function_shifts(shape[0], full_domain)
    r_values = shifts.astype(np.float64) * float(dx)

    s3_x = _compute_axis_third_order_fft(vx, 0, plan, local_shape, shape, box, comm, full_domain=full_domain)
    s3_y = _compute_axis_third_order_fft(vy, 1, plan, local_shape, shape, box, comm, full_domain=full_domain)
    s3_z = _compute_axis_third_order_fft(vz, 2, plan, local_shape, shape, box, comm, full_domain=full_domain)
    s3_avg = (s3_x + s3_y + s3_z) / 3.0
    return r_values, s3_x, s3_y, s3_z, s3_avg


def compute_shell_averaged_third_order_structure_function_fft(
    plan,
    local_shape,
    box,
    vx,
    vy,
    vz,
    shape,
    dx,
    dy,
    dz,
    comm,
):
    """Compute a binned lattice-shell average of the longitudinal third-order structure function."""
    _third_order_validate_grid(shape, dx, dy, dz)

    max_shift = int(shape[0] // 2)
    mx, my, mz = _local_shortest_periodic_displacements(shape, box)
    radius_sq = mx * mx + my * my + mz * mz
    radius = np.sqrt(radius_sq.astype(np.float64))

    canonical_mask = radius_sq > 0
    canonical_mask &= (mx > 0) | ((mx == 0) & (my > 0)) | ((mx == 0) & (my == 0) & (mz > 0))

    ex = np.zeros(local_shape, dtype=np.float64)
    ey = np.zeros(local_shape, dtype=np.float64)
    ez = np.zeros(local_shape, dtype=np.float64)
    ex[canonical_mask] = mx[canonical_mask] / radius[canonical_mask]
    ey[canonical_mask] = my[canonical_mask] / radius[canonical_mask]
    ez[canonical_mask] = mz[canonical_mask] / radius[canonical_mask]
    direction_cosines = (ex, ey, ez)

    components = (
        np.asarray(vx, dtype=np.float64),
        np.asarray(vy, dtype=np.float64),
        np.asarray(vz, dtype=np.float64),
    )
    component_ffts = tuple(
        forward_field(plan, component).reshape(local_shape, order="C")
        for component in components
    )
    pair_indices = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    quadratic_ffts = {
        (i, j): forward_field(plan, components[i] * components[j]).reshape(local_shape, order="C")
        for i, j in pair_indices
    }
    global_points = float(np.prod(shape))

    s3_local = np.zeros(local_shape, dtype=np.float64)
    for i, j in pair_indices:
        multiplicity = 1.0 if i == j else 2.0
        coeff_ij = multiplicity * direction_cosines[i] * direction_cosines[j]
        quad_fft = quadratic_ffts[(i, j)]
        for k in range(3):
            corr_local = backward_field(plan, np.conj(quad_fft) * component_ffts[k], local_shape) / global_points
            s3_local += 3.0 * coeff_ij * direction_cosines[k] * corr_local

    for j, k in pair_indices:
        multiplicity = 1.0 if j == k else 2.0
        coeff_jk = multiplicity * direction_cosines[j] * direction_cosines[k]
        quad_fft = quadratic_ffts[(j, k)]
        for i in range(3):
            corr_local = backward_field(plan, np.conj(component_ffts[i]) * quad_fft, local_shape) / global_points
            s3_local -= 3.0 * direction_cosines[i] * coeff_jk * corr_local

    local_shell_index = np.rint(radius[canonical_mask]).astype(np.int64, copy=False).ravel(order="C")
    local_s3_values = s3_local[canonical_mask].astype(np.float64, copy=False).ravel(order="C")
    valid_shell_mask = (local_shell_index > 0) & (local_shell_index <= max_shift)
    local_shell_index = local_shell_index[valid_shell_mask]
    local_s3_values = local_s3_values[valid_shell_mask]
    local_shell_sums = np.bincount(local_shell_index, weights=local_s3_values, minlength=max_shift + 1)
    local_shell_counts = np.bincount(local_shell_index, minlength=max_shift + 1).astype(np.float64)

    global_shell_sums = np.zeros_like(local_shell_sums) if comm.Get_rank() == 0 else None
    global_shell_counts = np.zeros_like(local_shell_counts) if comm.Get_rank() == 0 else None
    comm.Reduce(local_shell_sums, global_shell_sums, op=MPI.SUM, root=0)
    comm.Reduce(local_shell_counts, global_shell_counts, op=MPI.SUM, root=0)

    if comm.Get_rank() != 0:
        return None, None, None

    valid_shell_index = np.nonzero(global_shell_counts > 0.0)[0]
    valid_shell_index = valid_shell_index[valid_shell_index > 0]
    r_values = float(dx) * valid_shell_index.astype(np.float64)
    shell_average = global_shell_sums[valid_shell_index] / global_shell_counts[valid_shell_index]
    shell_counts = global_shell_counts[valid_shell_index]
    return r_values, shell_average, shell_counts


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
    bins = int(bins)
    if bins < 1:
        raise ValueError("Q-R joint PDF bin count must be at least 1.")

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
        zero_edges = np.linspace(-1.0, 1.0, bins + 1, dtype=np.float64)
        zero_centers = 0.5 * (zero_edges[:-1] + zero_edges[1:])
        zero_hist = np.zeros((bins, bins), dtype=np.float64)
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
            "bins": int(bins),
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

    q_edges = np.linspace(q_global_min, q_global_max, bins + 1, dtype=np.float64)
    r_edges = np.linspace(r_global_min, r_global_max, bins + 1, dtype=np.float64)

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
        "bins": int(bins),
        "q_min": float(q_global_min),
        "q_max": float(q_global_max),
        "r_min": float(r_global_min),
        "r_max": float(r_global_max),
    }
