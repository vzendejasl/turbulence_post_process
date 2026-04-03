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
from .io import save_spectra
from .io import structured_h5_metadata
from .layout import box_shape
from .layout import build_boxes
from .layout import choose_proc_grid
from .layout import scatter_field
from .spectra import compute_energy_dissipation_enstrophy
from .spectra import compute_energy_spectrum_from_modes
from .spectra import compute_enstrophy_spectrum_from_modes
from .spectra import compute_helicity_spectrum_from_modes
from .spectra import compensate_spectrum
from .transform import backward_field
from .transform import forward_field
from .transform import get_backend
from .transform import local_integer_wavenumber_mesh
from .transform import local_wavenumber_mesh
from .transform import print_component_ranges
from .transform import verify_decomposition


def analyze_file_parallel(filename, comm, header_lines=None, chunk_size=5_000_000, backend_name="heffte_fftw", visualize=False):
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

    if root:
        print()
        print("Performing HeFFTe Helmholtz-Hodge decomposition...")

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
    k_centers, E_total = compute_energy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, local_box, comm)
    _, E_comp = compute_energy_spectrum_from_modes(vx_c_k, vy_c_k, vz_c_k, shape, local_box, comm)
    _, E_rot = compute_energy_spectrum_from_modes(vx_r_k, vy_r_k, vz_r_k, shape, local_box, comm)
    _, Enst, total_enstrophy = compute_enstrophy_spectrum_from_modes(vx_k, vy_k, vz_k, shape, local_box, comm)
    _, Hel = compute_helicity_spectrum_from_modes(vx_k, vy_k, vz_k, shape, local_box, comm)

    KX_int, KY_int, KZ_int = local_integer_wavenumber_mesh(shape, local_box)
    omega_x_k = 1j * (KY_int * vz_k - KZ_int * vy_k)
    omega_y_k = 1j * (KZ_int * vx_k - KX_int * vz_k)
    omega_z_k = 1j * (KX_int * vy_k - KY_int * vx_k)
    omega_x = backward_field(plan, omega_x_k, local_shape)
    omega_y = backward_field(plan, omega_y_k, local_shape)
    omega_z = backward_field(plan, omega_z_k, local_shape)
    vorticity_ke = global_mean_energy(omega_x, omega_y, omega_z, global_points, comm)

    if root:
        print(f"  Total enstrophy (fourier, code convention): {total_enstrophy:.8f}")
        print(f"  Vorticity KE (real-space, code convention): {vorticity_ke:.8f}")

    if not np.isclose(vorticity_ke, total_enstrophy, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError(
            "Enstrophy sanity check failed: "
            f"vorticity KE = {vorticity_ke:.16e}, "
            f"enstrophy = {total_enstrophy:.16e}"
        )
    if root:
        print("  Sanity check: vorticity KE matches enstrophy.")

    compute_energy_dissipation_enstrophy(vx_k, vy_k, vz_k, shape, local_box, comm, root)

    result = None
    if root:
        E_total = zero_near_zero(E_total)
        E_comp = zero_near_zero(E_comp)
        E_rot = zero_near_zero(E_rot)
        Enst = zero_near_zero(Enst)
        Hel = zero_near_zero(Hel)
        E_total_comp = zero_near_zero(compensate_spectrum(k_centers, E_total, 5.0 / 3.0))
        E_comp_comp = zero_near_zero(compensate_spectrum(k_centers, E_comp, 5.0 / 3.0))
        E_rot_comp = zero_near_zero(compensate_spectrum(k_centers, E_rot, 5.0 / 3.0))
        Enst_comp = zero_near_zero(compensate_spectrum(k_centers, Enst, -1.0 / 3.0))
        total_ke = zero_near_zero_scalar(total_ke)
        comp_ke = zero_near_zero_scalar(comp_ke)
        rot_ke = zero_near_zero_scalar(rot_ke)
        total_enstrophy = zero_near_zero_scalar(total_enstrophy)
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
        if visualize:
            print("Visualization is not implemented in the parallel script yet.")
        result = {
            "k_centers": k_centers,
            "E_total": E_total,
            "E_comp": E_comp,
            "E_rot": E_rot,
            "Enstrophy": Enst,
            "Helicity": Hel,
            "E_total_compensated": E_total_comp,
            "E_comp_compensated": E_comp_comp,
            "E_rot_compensated": E_rot_comp,
            "Enstrophy_compensated": Enst_comp,
            "step_number": step_number,
            "time_value": time_value,
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
