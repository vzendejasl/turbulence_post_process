"""Application/CLI layer for distributed FFT post-processing."""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from .common import global_mean_energy
from .common import heffte
from .common import zero_near_zero
from .common import zero_near_zero_scalar
from .io import detect_header_lines
from .io import plot_spectra
from .io import read_data_file_chunked
from .io import read_data_file_header
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
from .transform import local_integer_wavenumber_mesh
from .transform import local_wavenumber_mesh
from .transform import print_component_ranges
from .transform import verify_decomposition


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

    header_lines = comm.bcast(header_lines, root=0)
    step_number = comm.bcast(step_number, root=0)
    time_value = comm.bcast(time_value, root=0)
    structured_h5 = comm.bcast(structured_h5, root=0)
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
        print("Decomposition component ranges:")

    print_component_ranges("Compressive component", vx_c, vy_c, vz_c, comm, root)
    print_component_ranges("Rotational component", vx_r, vy_r, vz_r, comm, root)

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
        comm,
    )
    Enst = None if Enst_x is None else Enst_x + Enst_y + Enst_z
    _, Hel = compute_helicity_spectrum_from_modes(vx_k, vy_k, vz_k, shape, local_box, comm)

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, local_box)
    omega_x_k = 1j * (KY_int * vz_k - KZ_int * vy_k)
    omega_y_k = 1j * (KZ_int * vx_k - KX_int * vz_k)
    omega_z_k = 1j * (KX_int * vy_k - KY_int * vx_k)
    omega_x = backward_field(plan, omega_x_k, local_shape)
    omega_y = backward_field(plan, omega_y_k, local_shape)
    omega_z = backward_field(plan, omega_z_k, local_shape)
    vorticity_ke = global_mean_energy(omega_x, omega_y, omega_z, global_points, comm)
    enstrophy_rel_error = abs(vorticity_ke - total_enstrophy) / max(abs(total_enstrophy), 1.0e-30)

    if root:
        print(f"  Total enstrophy (fourier, code convention): {total_enstrophy:.8f}")
        print(f"  Vorticity KE (real-space, code convention): {vorticity_ke:.8f}")
        print(f"  Enstrophy relative error: {enstrophy_rel_error:.8e}")

    compute_energy_dissipation_enstrophy(vx_k, vy_k, vz_k, shape, local_box, comm, root)

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
        r_values = None
        structure_function_longitudinal = None
        structure_function_path = None
        large_r_reference = None
        large_r_relative_difference = None
        E_total_comp = zero_near_zero(compensate_spectrum(k_centers, E_total, 5.0 / 3.0))
        E_comp_comp = zero_near_zero(compensate_spectrum(k_centers, E_comp, 5.0 / 3.0))
        E_rot_comp = zero_near_zero(compensate_spectrum(k_centers, E_rot, 5.0 / 3.0))
        Enst_comp = zero_near_zero(compensate_spectrum(k_centers, Enst, -1.0 / 3.0))
        total_ke = zero_near_zero_scalar(total_ke)
        comp_ke = zero_near_zero_scalar(comp_ke)
        rot_ke = zero_near_zero_scalar(rot_ke)
        total_enstrophy = zero_near_zero_scalar(total_enstrophy)
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
            E_total,
            E_comp,
            E_rot,
            Enst,
            Hel,
            E_total_comp,
            E_comp_comp,
            E_rot_comp,
            Enst_comp,
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
            "E_total": E_total,
            "E_total_x": E_total_x,
            "E_total_y": E_total_y,
            "E_total_z": E_total_z,
            "E_comp": E_comp,
            "E_rot": E_rot,
            "Enstrophy": Enst,
            "Enstrophy_x": Enst_x,
            "Enstrophy_y": Enst_y,
            "Enstrophy_z": Enst_z,
            "Helicity": Hel,
            "E_total_compensated": E_total_comp,
            "E_comp_compensated": E_comp_comp,
            "E_rot_compensated": E_rot_comp,
            "Enstrophy_compensated": Enst_comp,
            "step_number": step_number,
            "time_value": time_value,
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
