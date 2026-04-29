"""Application/CLI layer for distributed FFT post-processing."""

from __future__ import annotations

import argparse

import h5py
import numpy as np
from mpi4py import MPI

from .common import global_field_stats
from .common import global_mean
from .common import global_mean_energy
from .common import heffte
from .common import zero_near_zero
from .common import zero_near_zero_scalar
from .io import detect_header_lines
from .io import plot_spectra
from .io import read_data_file_chunked
from .io import read_data_file_header
from .io import read_structured_local_dataset
from .io import read_structured_local_fields
from .io import plot_qr_joint_pdf
from .io import save_component_spectra
from .io import save_qr_joint_pdf
from .io import save_structure_functions
from .io import save_spectra
from .io import structured_h5_metadata
from .layout import box_shape
from .layout import build_boxes
from .layout import choose_proc_grid
from .layout import scatter_field
from .spectra import compute_energy_dissipation_enstrophy
from .spectra import compute_energy_component_spectra_from_modes
from .spectra import compute_energy_spectrum_from_modes
from .spectra import compute_enstrophy_component_spectra_from_modes
from .spectra import compute_helicity_spectrum_from_modes
from .spectra import compute_shell_averaged_third_order_structure_function_fft
from .spectra import compute_third_order_structure_function_direct
from .spectra import compute_third_order_structure_function_fft
from .spectra import compute_longitudinal_structure_function_from_spectrum
from .spectra import compute_qr_joint_pdf
from .spectra import compensate_spectrum
from .transform import backward_field
from .transform import forward_field
from .transform import get_backend
from .transform import local_wavenumber_mesh
from .transform import print_component_ranges
from .transform import verify_decomposition


def _print_scalar_stats_block(label, stats):
    """Print a consistently formatted scalar-statistics block."""
    print(f"  {label}:")
    print(f"    min: {float(stats['global_min']):.8f}")
    print(f"    max: {float(stats['global_max']):.8f}")
    print(f"    rms: {float(stats['global_rms']):.8f}")
    print(f"    avg: {float(stats['global_mean']):.8f}")


def analyze_file_parallel(
    filename,
    comm,
    header_lines=None,
    chunk_size=5_000_000,
    backend_name="heffte_fftw",
    visualize=False,
    compute_structure_functions=False,
    structure_function_full_domain=True,
    qr_joint_pdf_bins=256,
    thermo_gamma=1.4,
):
    rank = comm.Get_rank()
    root = rank == 0
    structured_h5 = False

    if root:
        print()
        print(f"\n{'=' * 60}")
        print(f"ANALYZING: {filename}")
        print(f"{'=' * 60}")

        if filename.endswith(".h5"):
            header_lines = 0
            meta = structured_h5_metadata(filename)
            structured_h5 = meta is not None
        elif header_lines is None:
            header_lines = detect_header_lines(filename)

        step_number, time_value = read_data_file_header(filename, header_lines)
        has_thermo_inputs = False
        if structured_h5:
            with h5py.File(filename, "r") as hf:
                fields_group = hf.get("fields")
                has_thermo_inputs = (
                    fields_group is not None
                    and "density" in fields_group
                    and "pressure" in fields_group
                )
        if structured_h5:
            x_unique = meta["x_coords"]
            y_unique = meta["y_coords"]
            z_unique = meta["z_coords"]
            dx = meta["dx"]
            dy = meta["dy"]
            dz = meta["dz"]
            shape = meta["shape"]
            grid_vx = grid_vy = grid_vz = None
        else:
            grid_vx, grid_vy, grid_vz, x_unique, y_unique, z_unique, dx, dy, dz = read_data_file_chunked(
                filename, chunk_size=chunk_size, skiprows=header_lines
            )
            shape = (len(x_unique), len(y_unique), len(z_unique))
    else:
        step_number = None
        time_value = None
        grid_vx = grid_vy = grid_vz = None
        dx = dy = dz = None
        shape = None
        if filename.endswith(".h5"):
            header_lines = 0
        structured_h5 = None
        has_thermo_inputs = None

    header_lines = comm.bcast(header_lines, root=0)
    step_number = comm.bcast(step_number, root=0)
    time_value = comm.bcast(time_value, root=0)
    structured_h5 = comm.bcast(structured_h5, root=0)
    has_thermo_inputs = comm.bcast(has_thermo_inputs, root=0)
    shape = comm.bcast(shape, root=0)
    dx = comm.bcast(dx, root=0)
    dy = comm.bcast(dy, root=0)
    dz = comm.bcast(dz, root=0)

    proc_grid = choose_proc_grid(shape, comm.Get_size())
    boxes = build_boxes(shape, proc_grid)
    local_box = boxes[rank]
    local_shape = box_shape(local_box)
    global_points = int(np.prod(shape))

    if root:
        print(f"Using processor grid: {proc_grid}")
        print(f"Local box size on rank 0: {local_shape}")

    if structured_h5:
        if root:
            if h5py.get_config().mpi:
                print("Reading structured HDF5 slabs directly on each rank with MPI-enabled parallel HDF5...")
            else:
                print("Reading structured HDF5 slabs directly on each rank with serial h5py (independent reads)...")
        local_vx, local_vy, local_vz = read_structured_local_fields(filename, local_box, comm)
    else:
        if root and filename.endswith(".h5"):
            print("Legacy HDF5 read mode: serial read on rank 0 followed by MPI scatter.")
        local_vx_flat = scatter_field(grid_vx, boxes, comm)
        local_vy_flat = scatter_field(grid_vy, boxes, comm)
        local_vz_flat = scatter_field(grid_vz, boxes, comm)

        if root:
            del grid_vx, grid_vy, grid_vz

        local_vx = local_vx_flat.reshape(local_shape, order="C")
        local_vy = local_vy_flat.reshape(local_shape, order="C")
        local_vz = local_vz_flat.reshape(local_shape, order="C")

    total_ke = global_mean_energy(local_vx, local_vy, local_vz, global_points, comm)
    sound_speed_stats = None
    mach_number_stats = None
    turbulent_mach_number_stats = None
    turbulent_mach_fluctuation_stats = None
    mean_velocity_component_stats = None
    if has_thermo_inputs:
        if thermo_gamma <= 0.0:
            raise ValueError(f"Thermodynamic gamma must be positive. Received {thermo_gamma!r}.")

        local_density = read_structured_local_dataset(filename, "density", local_box, comm)
        local_pressure = read_structured_local_dataset(filename, "pressure", local_box, comm)
        invalid_density_count = comm.allreduce(int(np.count_nonzero(local_density <= 0.0)), op=MPI.SUM)
        if invalid_density_count:
            raise ValueError(
                f"Cannot compute thermodynamic diagnostics: found {invalid_density_count} "
                "non-positive density value(s)."
            )
        local_sound_speed_sq = thermo_gamma * local_pressure / local_density
        invalid_sound_speed_sq_count = comm.allreduce(
            int(np.count_nonzero(local_sound_speed_sq < 0.0)),
            op=MPI.SUM,
        )
        if invalid_sound_speed_sq_count:
            raise ValueError(
                f"Cannot compute thermodynamic diagnostics: found {invalid_sound_speed_sq_count} "
                "negative gamma*p/rho value(s)."
            )

        local_sound_speed = np.sqrt(np.maximum(local_sound_speed_sq, 0.0))
        local_speed = np.sqrt(local_vx**2 + local_vy**2 + local_vz**2)
        sound_speed_floor = np.maximum(local_sound_speed, 1.0e-30)
        sound_speed_stats = global_field_stats(local_sound_speed, comm)
        sound_speed_mean = float(sound_speed_stats["global_mean"])
        sound_speed_mean_floor = max(sound_speed_mean, 1.0e-30)

        turbulent_speed_scale = float(np.sqrt(max(2.0 * total_ke, 0.0)))
        mean_vx = float(global_mean(local_vx, comm))
        mean_vy = float(global_mean(local_vy, comm))
        mean_vz = float(global_mean(local_vz, comm))
        mean_velocity_component_stats = {
            "mean_vx": mean_vx,
            "mean_vy": mean_vy,
            "mean_vz": mean_vz,
            "mean_speed_magnitude": float(np.sqrt(mean_vx**2 + mean_vy**2 + mean_vz**2)),
        }
        fluctuation_speed_scale_sq = max(
            2.0 * total_ke - (mean_vx**2 + mean_vy**2 + mean_vz**2),
            0.0,
        )
        turbulent_fluctuation_speed_scale = float(np.sqrt(fluctuation_speed_scale_sq))
        turbulent_mach_value = turbulent_speed_scale / sound_speed_mean_floor
        turbulent_mach_fluctuation_value = turbulent_fluctuation_speed_scale / sound_speed_mean_floor

        mach_number_stats = global_field_stats(local_speed / sound_speed_floor, comm)
        turbulent_mach_number_stats = {
            "global_min": float(turbulent_mach_value),
            "global_max": float(turbulent_mach_value),
            "global_rms": float(turbulent_mach_value),
            "global_mean": float(turbulent_mach_value),
        }
        turbulent_mach_fluctuation_stats = {
            "global_min": float(turbulent_mach_fluctuation_value),
            "global_max": float(turbulent_mach_fluctuation_value),
            "global_rms": float(turbulent_mach_fluctuation_value),
            "global_mean": float(turbulent_mach_fluctuation_value),
        }

    KX, KY, KZ = local_wavenumber_mesh(shape, local_box, dx, dy, dz)
    K_squared = KX**2 + KY**2 + KZ**2
    nonzero_mask = K_squared > 0.0

    backend = get_backend(backend_name)
    plan = heffte.fft3d(backend, local_box, local_box, comm)
    third_order_plan = heffte.fft3d(backend, local_box, local_box, comm)

    if root:
        print()
        print("Performing HeFFTe Helmholtz-Hodge decomposition...")

    third_order_r_values = third_order_s3_x = third_order_s3_y = third_order_s3_z = third_order_s3_avg = None
    shell_r_values = shell_s3_values = shell_s3_counts = None
    if compute_structure_functions:
        third_order_r_values, third_order_s3_x, third_order_s3_y, third_order_s3_z, third_order_s3_avg = (
            compute_third_order_structure_function_fft(
                third_order_plan,
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
                full_domain=structure_function_full_domain,
            )
        )
        shell_result = compute_shell_averaged_third_order_structure_function_fft(
            third_order_plan,
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
        if root:
            shell_r_values, shell_s3_values, shell_s3_counts = shell_result
    vx_k = forward_field(plan, local_vx)
    vy_k = forward_field(plan, local_vy)
    vz_k = forward_field(plan, local_vz)

    vx_k = vx_k.reshape(local_shape, order="C")
    vy_k = vy_k.reshape(local_shape, order="C")
    vz_k = vz_k.reshape(local_shape, order="C")

    k_dot_v = KX * vx_k + KY * vy_k + KZ * vz_k
    projection = np.zeros_like(k_dot_v, dtype=np.complex128)
    projection[nonzero_mask] = k_dot_v[nonzero_mask] / K_squared[nonzero_mask]

    vx_c_k = KX * projection
    vy_c_k = KY * projection
    vz_c_k = KZ * projection

    vx_r_k = vx_k - vx_c_k
    vy_r_k = vy_k - vy_c_k
    vz_r_k = vz_k - vz_c_k

    vx_c = backward_field(plan, vx_c_k, local_shape)
    vy_c = backward_field(plan, vy_c_k, local_shape)
    vz_c = backward_field(plan, vz_c_k, local_shape)

    vx_r = backward_field(plan, vx_r_k, local_shape)
    vy_r = backward_field(plan, vy_r_k, local_shape)
    vz_r = backward_field(plan, vz_r_k, local_shape)

    comp_ke = global_mean_energy(vx_c, vy_c, vz_c, global_points, comm)
    rot_ke = global_mean_energy(vx_r, vy_r, vz_r, global_points, comm)

    if root:
        print()
        comp_pct = 100.0 * comp_ke / total_ke if total_ke > 0.0 else 0.0
        rot_pct = 100.0 * rot_ke / total_ke if total_ke > 0.0 else 0.0
        print("Energy breakdown:")
        print(f"  Total: {total_ke:.8f}")
        print(f"  Compressive: {comp_ke:.8f} ({comp_pct:.1f}%)")
        print(f"  Rotational: {rot_ke:.8f} ({rot_pct:.1f}%)")
        print(f"  Sum: {comp_ke + rot_ke:.8f}")
        print()
        print("Decomposition component ranges:")

    print_component_ranges("Compressive component", vx_c, vy_c, vz_c, comm, root)
    print_component_ranges("Rotational component", vx_r, vy_r, vz_r, comm, root)

    if root:
        print()
    verify_decomposition(plan, local_shape, KX, KY, KZ, vx_c_k, vy_c_k, vz_c_k, vx_r_k, vy_r_k, vz_r_k, comm, root)

    if root:
        print()
        print("Computing distributed energy spectra...")
    k_centers, E_total_x, E_total_y, E_total_z = compute_energy_component_spectra_from_modes(
        vx_k,
        vy_k,
        vz_k,
        shape,
        local_box,
        comm,
    )
    E_total = None if E_total_x is None else E_total_x + E_total_y + E_total_z
    _, E_comp = compute_energy_spectrum_from_modes(vx_c_k, vy_c_k, vz_c_k, shape, local_box, comm)
    _, E_rot = compute_energy_spectrum_from_modes(vx_r_k, vy_r_k, vz_r_k, shape, local_box, comm)
    _, Enst_x, Enst_y, Enst_z, total_enstrophy = compute_enstrophy_component_spectra_from_modes(
        vx_k,
        vy_k,
        vz_k,
        shape,
        local_box,
        dx,
        dy,
        dz,
        comm,
    )
    Enst = None if Enst_x is None else Enst_x + Enst_y + Enst_z
    _, Hel = compute_helicity_spectrum_from_modes(vx_k, vy_k, vz_k, shape, local_box, dx, dy, dz, comm)

    omega_x_k = 1j * (KY * vz_k - KZ * vy_k)
    omega_y_k = 1j * (KZ * vx_k - KX * vz_k)
    omega_z_k = 1j * (KX * vy_k - KY * vx_k)
    omega_x = backward_field(plan, omega_x_k, local_shape)
    omega_y = backward_field(plan, omega_y_k, local_shape)
    omega_z = backward_field(plan, omega_z_k, local_shape)
    vorticity_ke = global_mean_energy(omega_x, omega_y, omega_z, global_points, comm)
    wiwi_mean = 2.0 * vorticity_ke
    enstrophy_rel_error = abs(vorticity_ke - total_enstrophy) / max(abs(total_enstrophy), 1.0e-30)

    if root:
        print(f"  Total enstrophy (fourier): {total_enstrophy:.8f}")
        print(f"  Vorticity KE (real-space): {vorticity_ke:.8f}")
        print(f"  Enstrophy relative error: {enstrophy_rel_error:.8e}")

    compute_energy_dissipation_enstrophy(vx_k, vy_k, vz_k, shape, local_box, dx, dy, dz, comm, root)

    if root and sound_speed_stats is not None:
        print()
        print("Thermodynamic diagnostics:")
        print(f"  gamma: {float(thermo_gamma):.6g}")
        print()
        _print_scalar_stats_block("Sound speed", sound_speed_stats)
        print()
        _print_scalar_stats_block("Mach number", mach_number_stats)
        print()
        print("  Mean velocity components:")
        print(f"    <u> = {float(mean_velocity_component_stats['mean_vx']):.8e}")
        print(f"    <v> = {float(mean_velocity_component_stats['mean_vy']):.8e}")
        print(f"    <w> = {float(mean_velocity_component_stats['mean_vz']):.8e}")
        print(
            "    |<u_i>| = "
            f"{float(mean_velocity_component_stats['mean_speed_magnitude']):.8e}"
        )
        print()
        print("  Turbulent Mach number:")
        print(
            "    Mt_raw = sqrt(2<KE>) / c_mean = "
            f"{float(turbulent_mach_number_stats['global_mean']):.8f}"
        )
        print(
            "    Mt_fluct = u'_rms / c_mean = "
            f"{float(turbulent_mach_fluctuation_stats['global_mean']):.8f}"
        )

    if root:
        print()
        print("Computing distributed Q-R joint PDF...")
        print("  Normalization: q_A = Q / |grad(u)|_F^2, r_A = R / |grad(u)|_F^3")
        print("  Filter: |grad(u)|_F^2 / max(|grad(u)|_F^2) >= 1e-3")

    dux_dx = backward_field(plan, 1j * KX * vx_k, local_shape)
    dux_dy = backward_field(plan, 1j * KY * vx_k, local_shape)
    dux_dz = backward_field(plan, 1j * KZ * vx_k, local_shape)
    duy_dx = backward_field(plan, 1j * KX * vy_k, local_shape)
    duy_dy = backward_field(plan, 1j * KY * vy_k, local_shape)
    duy_dz = backward_field(plan, 1j * KZ * vy_k, local_shape)
    duz_dx = backward_field(plan, 1j * KX * vz_k, local_shape)
    duz_dy = backward_field(plan, 1j * KY * vz_k, local_shape)
    duz_dz = backward_field(plan, 1j * KZ * vz_k, local_shape)

    s11 = dux_dx
    s22 = duy_dy
    s33 = duz_dz
    s12 = 0.5 * (dux_dy + duy_dx)
    s13 = 0.5 * (dux_dz + duz_dx)
    s23 = 0.5 * (duy_dz + duz_dy)
    local_sijsij = np.sum(
        s11**2 + s22**2 + s33**2 + 2.0 * (s12**2 + s13**2 + s23**2),
        dtype=np.float64,
    )
    sijsij_mean = comm.allreduce(local_sijsij, op=MPI.SUM) / float(global_points)
    strain_enstrophy_rel_error = abs(sijsij_mean - total_enstrophy) / max(abs(total_enstrophy), 1.0e-30)
    strain_vorticity_rel_error = abs(2.0 * sijsij_mean - wiwi_mean) / max(abs(wiwi_mean), 1.0e-30)

    if root:
        print()
        print("Strain-vorticity diagnostic:")
        print(f"  <SijSij>: {sijsij_mean:.8f}")
        print(f"  <wiwi>: {wiwi_mean:.8f}")
        print(f"  2 * <SijSij>: {2.0 * sijsij_mean:.8f}")
        print(f"  Relative error in <SijSij> vs enstrophy: {strain_enstrophy_rel_error:.8e}")
        print(f"  Relative error in 2<SijSij> vs <wiwi>: {strain_vorticity_rel_error:.8e}")

    qr_joint_pdf = compute_qr_joint_pdf(
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
        bins=qr_joint_pdf_bins,
    )

    result = None
    if root:
        E_total = zero_near_zero(E_total)
        E_total_x = zero_near_zero(E_total_x)
        E_total_y = zero_near_zero(E_total_y)
        E_total_z = zero_near_zero(E_total_z)
        E_comp = zero_near_zero(E_comp)
        E_rot = zero_near_zero(E_rot)
        Enst = zero_near_zero(Enst)
        Enst_x = zero_near_zero(Enst_x)
        Enst_y = zero_near_zero(Enst_y)
        Enst_z = zero_near_zero(Enst_z)
        Hel = zero_near_zero(Hel)
        domain_length = float(dx) * float(shape[0])
        delta_k_phy = float(2.0 * np.pi / domain_length)
        k_centers_phy = zero_near_zero((2.0 * np.pi / domain_length) * np.asarray(k_centers, dtype=np.float64))
        E_total_phy_density = zero_near_zero(np.asarray(E_total, dtype=np.float64) / delta_k_phy)
        E_comp_phy_density = zero_near_zero(np.asarray(E_comp, dtype=np.float64) / delta_k_phy)
        E_rot_phy_density = zero_near_zero(np.asarray(E_rot, dtype=np.float64) / delta_k_phy)
        Enst_phy_density = zero_near_zero(np.asarray(Enst, dtype=np.float64) / delta_k_phy)
        Hel_phy_density = zero_near_zero(np.asarray(Hel, dtype=np.float64) / delta_k_phy)
        r_values = None
        structure_function_longitudinal = None
        structure_function_path = None
        large_r_reference = None
        large_r_relative_difference = None
        E_total_comp = zero_near_zero(compensate_spectrum(k_centers, E_total, 5.0 / 3.0))
        E_total_comp_phy = zero_near_zero(compensate_spectrum(k_centers_phy, E_total_phy_density, 5.0 / 3.0))
        E_comp_comp = zero_near_zero(compensate_spectrum(k_centers, E_comp, 5.0 / 3.0))
        E_comp_comp_phy = zero_near_zero(compensate_spectrum(k_centers_phy, E_comp_phy_density, 5.0 / 3.0))
        E_rot_comp = zero_near_zero(compensate_spectrum(k_centers, E_rot, 5.0 / 3.0))
        E_rot_comp_phy = zero_near_zero(compensate_spectrum(k_centers_phy, E_rot_phy_density, 5.0 / 3.0))
        Enst_comp = zero_near_zero(compensate_spectrum(k_centers, Enst, -1.0 / 3.0))
        Enst_comp_phy = zero_near_zero(compensate_spectrum(k_centers_phy, Enst_phy_density, -1.0 / 3.0))
        total_ke = zero_near_zero_scalar(total_ke)
        comp_ke = zero_near_zero_scalar(comp_ke)
        rot_ke = zero_near_zero_scalar(rot_ke)
        total_enstrophy = zero_near_zero_scalar(total_enstrophy)
        wiwi_mean = zero_near_zero_scalar(wiwi_mean)
        sijsij_mean = zero_near_zero_scalar(sijsij_mean)
        strain_enstrophy_rel_error = zero_near_zero_scalar(strain_enstrophy_rel_error)
        strain_vorticity_rel_error = zero_near_zero_scalar(strain_vorticity_rel_error)
        third_order_result = None
        if compute_structure_functions:
            max_r_index = shape[0] if structure_function_full_domain else shape[0] // 2
            r_values = np.arange(max_r_index + 1, dtype=np.float64) * float(dx)
            _, structure_function_longitudinal = compute_longitudinal_structure_function_from_spectrum(
                k_centers,
                E_total,
                r_values,
                domain_length,
            )
            structure_function_longitudinal = zero_near_zero(structure_function_longitudinal)
            large_r_reference = (4.0 / 3.0) * float(np.sum(E_total, dtype=np.float64))
            large_r_reference_index = shape[0] // 2
            large_r_reference_r = float(r_values[large_r_reference_index])
            large_r_relative_difference = abs(
                float(structure_function_longitudinal[large_r_reference_index]) - large_r_reference
            ) / max(abs(large_r_reference), 1.0e-30)
            print("Longitudinal structure function diagnostic:")
            print(f"  Largest saved r: {float(r_values[-1]):.8f}")
            print(f"  Reference comparison r: {large_r_reference_r:.8f}")
            print(f"  S_L(r_ref): {float(structure_function_longitudinal[large_r_reference_index]):.8f}")
            print(f"  4/3 * sum(E_total): {large_r_reference:.8f}")
            print(f"  Relative difference at r_ref: {large_r_relative_difference:.8e}")
            third_order_s3_x = zero_near_zero(third_order_s3_x)
            third_order_s3_y = zero_near_zero(third_order_s3_y)
            third_order_s3_z = zero_near_zero(third_order_s3_z)
            third_order_s3_avg = zero_near_zero(third_order_s3_avg)
            directional_spread_abs = np.maximum.reduce(
                (
                    np.abs(third_order_s3_x - third_order_s3_avg),
                    np.abs(third_order_s3_y - third_order_s3_avg),
                    np.abs(third_order_s3_z - third_order_s3_avg),
                )
            )
            max_abs_directional_spread_index = int(np.argmax(directional_spread_abs))
            max_abs_directional_spread = float(directional_spread_abs[max_abs_directional_spread_index])
            r_at_max_abs_directional_spread = float(third_order_r_values[max_abs_directional_spread_index])
            directional_spread_rel = np.full_like(directional_spread_abs, np.nan, dtype=np.float64)
            valid_rel_mask = np.abs(third_order_s3_avg) > 1.0e-30
            directional_spread_rel[valid_rel_mask] = (
                directional_spread_abs[valid_rel_mask] / np.abs(third_order_s3_avg[valid_rel_mask])
            )
            if np.any(valid_rel_mask):
                max_rel_directional_spread_index = int(np.nanargmax(directional_spread_rel))
                max_rel_directional_spread = float(directional_spread_rel[max_rel_directional_spread_index])
                r_at_max_rel_directional_spread = float(third_order_r_values[max_rel_directional_spread_index])
            else:
                max_rel_directional_spread = 0.0
                r_at_max_rel_directional_spread = 0.0
            third_order_direct_max_abs_diff = None
            if comm.Get_size() == 1:
                _, direct_s3_x, direct_s3_y, direct_s3_z, direct_s3_avg = compute_third_order_structure_function_direct(
                    local_vx,
                    local_vy,
                    local_vz,
                    dx,
                    dy,
                    dz,
                    full_domain=structure_function_full_domain,
                )
                third_order_direct_max_abs_diff = float(
                    max(
                        np.max(np.abs(third_order_s3_x - direct_s3_x)),
                        np.max(np.abs(third_order_s3_y - direct_s3_y)),
                        np.max(np.abs(third_order_s3_z - direct_s3_z)),
                        np.max(np.abs(third_order_s3_avg - direct_s3_avg)),
                    )
                )
            print("Third-order structure function diagnostic:")
            print(f"  Largest saved r: {float(third_order_r_values[-1]):.8f}")
            print(f"  S3_avg(r_max): {float(third_order_s3_avg[-1]):.8f}")
            print(
                f"  Max absolute directional spread: {max_abs_directional_spread:.8e} "
                f"at r = {r_at_max_abs_directional_spread:.8f}"
            )
            print(
                f"  Max relative directional spread: {max_rel_directional_spread:.8e} "
                f"at r = {r_at_max_rel_directional_spread:.8f}"
            )
            shell_s3_values = zero_near_zero(shell_s3_values)
            populated_shell_bins = int(np.count_nonzero(shell_s3_counts > 0.0))
            print(f"  Shell-average populated bins: {populated_shell_bins:d}")
            if third_order_direct_max_abs_diff is not None:
                print(f"  Max |FFT - direct| on 1 rank: {third_order_direct_max_abs_diff:.8e}")
            third_order_result = {
                "r_values": third_order_r_values,
                "S3_x": third_order_s3_x,
                "S3_y": third_order_s3_y,
                "S3_z": third_order_s3_z,
                "S3_avg": third_order_s3_avg,
                "direct_max_abs_diff": third_order_direct_max_abs_diff,
                "max_abs_directional_spread": max_abs_directional_spread,
                "r_at_max_abs_directional_spread": r_at_max_abs_directional_spread,
                "max_rel_directional_spread": max_rel_directional_spread,
                "r_at_max_rel_directional_spread": r_at_max_rel_directional_spread,
            }
        save_spectra(
            k_centers,
            k_centers_phy,
            E_total,
            E_total_phy_density,
            E_comp,
            E_comp_phy_density,
            E_rot,
            E_rot_phy_density,
            Enst,
            Enst_phy_density,
            Hel,
            Hel_phy_density,
            E_total_comp,
            E_total_comp_phy,
            E_comp_comp,
            E_comp_comp_phy,
            E_rot_comp,
            E_rot_comp_phy,
            Enst_comp,
            Enst_comp_phy,
            filename,
            step_number,
            time_value,
            shape[0],
            shape[1],
            shape[2],
            total_ke,
            comp_ke,
            rot_ke,
            total_enstrophy,
            sijsij_mean=sijsij_mean,
            wiwi_mean=wiwi_mean,
            strain_enstrophy_rel_error=strain_enstrophy_rel_error,
            strain_vorticity_rel_error=strain_vorticity_rel_error,
            thermo_gamma=thermo_gamma if sound_speed_stats is not None else None,
            sound_speed_stats=sound_speed_stats,
            mach_number_stats=mach_number_stats,
            turbulent_mach_number_stats=turbulent_mach_number_stats,
            turbulent_mach_fluctuation_stats=turbulent_mach_fluctuation_stats,
            mean_velocity_component_stats=mean_velocity_component_stats,
        )
        save_component_spectra(
            k_centers,
            E_total,
            E_total_x,
            E_total_y,
            E_total_z,
            Enst,
            Enst_x,
            Enst_y,
            Enst_z,
            filename,
        )
        if compute_structure_functions:
            structure_function_path = save_structure_functions(
                r_values,
                structure_function_longitudinal,
                third_order_s3_x,
                third_order_s3_y,
                third_order_s3_z,
                third_order_s3_avg,
                shell_r_values,
                shell_s3_values,
                shell_s3_counts,
                filename,
                step_number,
                time_value,
                domain_length,
                large_r_reference,
                large_r_relative_difference,
                large_r_reference_r,
                max_abs_directional_spread,
                r_at_max_abs_directional_spread,
                max_rel_directional_spread,
                r_at_max_rel_directional_spread,
                "full-box" if structure_function_full_domain else "half-box",
            )
            third_order_result["path"] = structure_function_path
            third_order_result["shell_r_values"] = shell_r_values
            third_order_result["S3_shell"] = shell_s3_values
            third_order_result["shell_count"] = shell_s3_counts
        qr_h5_path = save_qr_joint_pdf(
            qr_joint_pdf,
            filename,
            step_number,
            time_value,
            shape[0],
            shape[1],
            shape[2],
        )
        qr_pdf_path = plot_qr_joint_pdf(qr_joint_pdf, filename)
        if visualize:
            print("Visualization is not implemented in the parallel script yet.")
        result = {
            "k_centers": k_centers,
            "k_centers_phy": k_centers_phy,
            "E_total": E_total,
            "E_total_phy": E_total_phy_density,
            "E_total_x": E_total_x,
            "E_total_y": E_total_y,
            "E_total_z": E_total_z,
            "E_comp": E_comp,
            "E_comp_phy": E_comp_phy_density,
            "E_rot": E_rot,
            "E_rot_phy": E_rot_phy_density,
            "Enstrophy": Enst,
            "Enstrophy_phy": Enst_phy_density,
            "Enstrophy_x": Enst_x,
            "Enstrophy_y": Enst_y,
            "Enstrophy_z": Enst_z,
            "Helicity": Hel,
            "Helicity_phy": Hel_phy_density,
            "E_total_compensated": E_total_comp,
            "E_total_compensated_phy": E_total_comp_phy,
            "E_comp_compensated": E_comp_comp,
            "E_comp_compensated_phy": E_comp_comp_phy,
            "E_rot_compensated": E_rot_comp,
            "E_rot_compensated_phy": E_rot_comp_phy,
            "Enstrophy_compensated": Enst_comp,
            "Enstrophy_compensated_phy": Enst_comp_phy,
            "step_number": step_number,
            "time_value": time_value,
            "SijSij_mean": sijsij_mean,
            "wiwi_mean": wiwi_mean,
            "strain_enstrophy_rel_error": strain_enstrophy_rel_error,
            "strain_vorticity_rel_error": strain_vorticity_rel_error,
            "thermo_gamma": float(thermo_gamma) if sound_speed_stats is not None else None,
            "sound_speed_stats": sound_speed_stats,
            "mach_number_stats": mach_number_stats,
            "turbulent_mach_number_stats": turbulent_mach_number_stats,
            "turbulent_mach_fluctuation_stats": turbulent_mach_fluctuation_stats,
            "mean_velocity_component_stats": mean_velocity_component_stats,
            "r_values": r_values,
            "S_L": structure_function_longitudinal,
            "structure_function_path": structure_function_path,
            "S_L_large_r_reference": large_r_reference,
            "S_L_large_r_relative_difference": large_r_relative_difference,
            "third_order_structure_function": third_order_result,
            "qr_joint_pdf_h5": qr_h5_path,
            "qr_joint_pdf_pdf": qr_pdf_path,
            "qr_joint_pdf_bins": int(qr_joint_pdf_bins),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Distributed HeFFTe Helmholtz-Hodge decomposition and spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mpirun -n 4 python ComputeSpectra.py data_file.h5 --backend heffte_fftw --no-plot
  mpirun -n 8 python ComputeSpectra.py file1.txt file2.txt --header-lines 5
        """,
    )
    parser.add_argument("data_files", type=str, nargs="+", help="One or more velocity data files to analyze")
    parser.add_argument(
        "--header-lines",
        type=int,
        default=None,
        help="Number of header lines to skip/read. If omitted, attempts auto-detection on rank 0.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="heffte_fftw",
        choices=["heffte_fftw", "heffte_stock"],
        help="HeFFTe backend to use internally. Default is HeFFTe with FFTW.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Chunk size used when rank 0 reads the input file.",
    )
    parser.add_argument(
        "--qr-bins",
        type=int,
        default=256,
        help="Number of linear bins per axis for the Q-R joint PDF. Default is 256.",
    )
    parser.add_argument(
        "--structure-functions",
        action="store_true",
        help="Compute 2nd- and 3rd-order structure functions in addition to spectra.",
    )
    parser.add_argument(
        "--structure-function-full-box",
        dest="structure_function_full_box",
        action="store_true",
        help="Sample axis-aligned structure functions over the full periodic box (r = 0..L). This is the default.",
    )
    parser.add_argument(
        "--structure-function-half-box",
        dest="structure_function_full_box",
        action="store_false",
        help="Sample axis-aligned structure functions over only the shortest periodic half-box (r = 0..L/2).",
    )
    parser.set_defaults(structure_function_full_box=True)
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.4,
        help="Ideal-gas ratio of specific heats used for thermodynamic diagnostics when density and pressure exist.",
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="Reserved for future use.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting spectra on rank 0.")
    args = parser.parse_args()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    results = []
    for filename in args.data_files:
        result = analyze_file_parallel(
            filename,
            comm,
            header_lines=args.header_lines,
            chunk_size=args.chunk_size,
            backend_name=args.backend,
            visualize=args.visualize,
            compute_structure_functions=args.structure_functions,
            structure_function_full_domain=args.structure_function_full_box,
            qr_joint_pdf_bins=args.qr_bins,
            thermo_gamma=args.gamma,
        )
        if rank == 0:
            results.append(result)
        comm.Barrier()

    if rank == 0:
        if not args.no_plot:
            plot_spectra(results)
        else:
            print(f"\nProcessed {len(results)} files. Spectrum files saved to disk.")
            print("Skipping plot display as requested (--no-plot flag).")
