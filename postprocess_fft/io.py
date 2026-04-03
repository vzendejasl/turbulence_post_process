"""Input/output helpers for FFT post-processing."""

from __future__ import annotations

import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from convert_txt_to_hdf5 import is_structured_velocity_hdf5

from .layout import box_slices


def detect_header_lines(filename):
    """Count non-numeric header lines at the start of a text file."""
    header_count = 0
    with open(filename, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                header_count += 1
                continue
            try:
                [float(x) for x in stripped.split()]
                break
            except ValueError:
                header_count += 1
    return header_count


def read_data_file_header(filename, header_lines):
    """Extract step and time metadata from text or HDF5 input."""
    if filename.endswith(".h5"):
        with h5py.File(filename, "r") as hf:
            step_number = str(hf.attrs.get("step", "unknown"))
            time_value = float(hf.attrs.get("time", 0.0))
        return step_number, time_value

    step_number = "unknown"
    time_value = 0.0
    with open(filename, "r", encoding="utf-8") as handle:
        for _ in range(header_lines):
            line = handle.readline()
            if "Cycle" in line or "Step" in line:
                match = re.search(r"(?:Cycle|Step)\s*[:=]\s*(\d+)", line)
                if match:
                    step_number = match.group(1)
            if "Time" in line:
                match = re.search(r"Time\s*[:=]\s*([0-9.eE+-]+)", line)
                if match:
                    time_value = float(match.group(1))
    return step_number, time_value


def read_data_file_chunked(filename, chunk_size=5_000_000, skiprows=0):
    """Read a text velocity file and reconstruct structured 3D arrays."""
    chunks = []
    reader = pd.read_csv(
        filename,
        skiprows=skiprows,
        header=None,
        sep=r"\s+",
        chunksize=chunk_size,
    )
    for chunk_df in reader:
        chunks.append(chunk_df.values.astype(np.float64))

    if not chunks:
        raise ValueError(f"No data rows found in {filename}.")

    data = np.vstack(chunks)
    x_unique = np.unique(np.round(data[:, 0], 10))
    y_unique = np.unique(np.round(data[:, 1], 10))
    z_unique = np.unique(np.round(data[:, 2], 10))
    dx = float(x_unique[1] - x_unique[0]) if len(x_unique) > 1 else 1.0
    dy = float(y_unique[1] - y_unique[0]) if len(y_unique) > 1 else 1.0
    dz = float(z_unique[1] - z_unique[0]) if len(z_unique) > 1 else 1.0

    if len(x_unique) > 1:
        ix = np.rint((np.round(data[:, 0], 10) - x_unique[0]) / dx).astype(np.int64)
    else:
        ix = np.zeros(len(data), dtype=np.int64)
    if len(y_unique) > 1:
        iy = np.rint((np.round(data[:, 1], 10) - y_unique[0]) / dy).astype(np.int64)
    else:
        iy = np.zeros(len(data), dtype=np.int64)
    if len(z_unique) > 1:
        iz = np.rint((np.round(data[:, 2], 10) - z_unique[0]) / dz).astype(np.int64)
    else:
        iz = np.zeros(len(data), dtype=np.int64)

    shape = (len(x_unique), len(y_unique), len(z_unique))
    grid_vx = np.zeros(shape, dtype=np.float64)
    grid_vy = np.zeros(shape, dtype=np.float64)
    grid_vz = np.zeros(shape, dtype=np.float64)
    grid_vx[ix, iy, iz] = data[:, 3]
    grid_vy[ix, iy, iz] = data[:, 4]
    grid_vz[ix, iy, iz] = data[:, 5]

    return grid_vx, grid_vy, grid_vz, x_unique, y_unique, z_unique, dx, dy, dz


def open_h5_for_parallel_read(filename, comm):
    """Use collective MPI-IO when available, otherwise independent reads."""
    if h5py.get_config().mpi:
        return h5py.File(filename, "r", driver="mpio", comm=comm)
    return h5py.File(filename, "r")


def structured_h5_metadata(filename):
    """Return metadata for the FFT-ready HDF5 schema, or None for legacy files."""
    with h5py.File(filename, "r") as hf:
        if not is_structured_velocity_hdf5(hf):
            return None

        x_full = np.asarray(hf["grid"]["x"][:], dtype=np.float64)
        y_full = np.asarray(hf["grid"]["y"][:], dtype=np.float64)
        z_full = np.asarray(hf["grid"]["z"][:], dtype=np.float64)
        periodic_duplicate_last = bool(hf.attrs.get("periodic_duplicate_last", True))

    if periodic_duplicate_last and len(x_full) > 1 and len(y_full) > 1 and len(z_full) > 1:
        x_coords = x_full[:-1]
        y_coords = y_full[:-1]
        z_coords = z_full[:-1]
    else:
        x_coords = x_full
        y_coords = y_full
        z_coords = z_full

    dx = x_full[1] - x_full[0] if len(x_full) > 1 else 1.0
    dy = y_full[1] - y_full[0] if len(y_full) > 1 else 1.0
    dz = z_full[1] - z_full[0] if len(z_full) > 1 else 1.0

    return {
        "shape": (len(x_coords), len(y_coords), len(z_coords)),
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


def read_structured_local_fields(filename, local_box, comm):
    """Read one rank-local structured HDF5 slab directly from disk."""
    sx, sy, sz = box_slices(local_box)
    with open_h5_for_parallel_read(filename, comm) as hf:
        vx = np.asarray(hf["fields"]["vx"][sx, sy, sz], dtype=np.float64)
        vy = np.asarray(hf["fields"]["vy"][sx, sy, sz], dtype=np.float64)
        vz = np.asarray(hf["fields"]["vz"][sx, sy, sz], dtype=np.float64)
    return vx, vy, vz


def _spectra_output_stem(filename, step_number):
    directory = os.path.dirname(os.path.abspath(filename))
    base = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(directory, f"{base}_step{step_number}_spectra")


def save_spectra(
    k_centers,
    e_total,
    e_comp,
    e_rot,
    enstrophy,
    helicity,
    e_total_compensated,
    e_comp_compensated,
    e_rot_compensated,
    enstrophy_compensated,
    filename,
    step_number,
    time_value,
    nx,
    ny,
    nz,
    total_ke,
    comp_ke,
    rot_ke,
    total_enstrophy,
):
    """Save spectra text plus a separate metadata text file."""
    stem = _spectra_output_stem(filename, step_number)

    summary = np.column_stack(
        (
            np.asarray(k_centers, dtype=np.float64),
            np.asarray(e_total, dtype=np.float64),
            np.asarray(e_comp, dtype=np.float64),
            np.asarray(e_rot, dtype=np.float64),
            np.asarray(enstrophy, dtype=np.float64),
            np.asarray(helicity, dtype=np.float64),
            np.asarray(e_total_compensated, dtype=np.float64),
            np.asarray(e_comp_compensated, dtype=np.float64),
            np.asarray(e_rot_compensated, dtype=np.float64),
            np.asarray(enstrophy_compensated, dtype=np.float64),
        )
    )

    header_labels = [
        "k",
        "E_total",
        "E_comp",
        "E_rot",
        "Enstrophy",
        "Helicity",
        "E_total_comp",
        "E_comp_comp",
        "E_rot_comp",
        "Enst_comp",
    ]
    with open(f"{stem}.txt", "w", encoding="utf-8") as handle:
        handle.write(", ".join(f"{label:>23s}" for label in header_labels) + "\n")
        for row in summary:
            handle.write(", ".join(f"{value:>23.16e}" for value in row) + "\n")

    with open(f"{stem}_metadata.txt", "w", encoding="utf-8") as handle:
        handle.write(f"# Step: {step_number}, Time: {float(time_value):.16e}\n")
        handle.write(f"# Grid: Nx={int(nx)}, Ny={int(ny)}, Nz={int(nz)}\n")
        handle.write(
            f"# Total KE: {float(total_ke):.16e}, "
            f"Compressive KE: {float(comp_ke):.16e}, "
            f"Rotational KE: {float(rot_ke):.16e}\n"
        )
        handle.write(f"# Total Enstrophy: {float(total_enstrophy):.16e}\n")


def _plot_style():
    style = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
    return style


def plot_spectra(results):
    """Plot raw and compensated spectra for one or more results."""
    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        raw_ax, comp_ax = axes
        for result in results:
            k_centers = result["k_centers"]
            step_number = result["step_number"]
            time_value = result["time_value"]
            label = rf"$\mathrm{{step}}={step_number},\ t={time_value:.6g}$"
            raw_ax.loglog(k_centers, result["E_total"], label=rf"$E(k)$, {label}")
            raw_ax.loglog(k_centers, result["E_comp"], linestyle="--", label=rf"$E_c(k)$, {label}")
            raw_ax.loglog(k_centers, result["E_rot"], linestyle=":", label=rf"$E_r(k)$, {label}")
            raw_ax.loglog(k_centers, result["Enstrophy"], linestyle="-.", label=rf"$\Omega(k)$, {label}")

            comp_ax.semilogx(k_centers, result["E_comp_compensated"], linestyle="--", label=rf"$k^{{5/3}}E_c(k)$, {label}")
            comp_ax.semilogx(k_centers, result["E_rot_compensated"], linestyle=":", label=rf"$k^{{5/3}}E_r(k)$, {label}")
            comp_ax.semilogx(k_centers, result["Enstrophy_compensated"], linestyle="-.", label=rf"$k^{{-1/3}}\Omega(k)$, {label}")

        raw_ax.set_xlabel(r"$k$")
        raw_ax.set_ylabel(r"$E(k),\ \Omega(k)$")
        raw_ax.grid(True, which="both")
        raw_ax.legend(fontsize=8)

        comp_ax.set_xlabel(r"$k$")
        comp_ax.set_ylabel(r"Compensated spectra")
        comp_ax.grid(True, which="both")
        comp_ax.legend(fontsize=8)

        fig.tight_layout()
        plt.show()
