"""Input/output helpers for FFT post-processing."""

from __future__ import annotations

import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postprocess_lib.converter import is_structured_velocity_hdf5

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


def _spectra_output_stem(filename):
    directory = os.path.dirname(os.path.abspath(filename))
    base = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(directory, f"{base}_spectra")


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
    stem = _spectra_output_stem(filename)

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


def save_qr_joint_pdf(
    qr_result,
    filename,
    step_number,
    time_value,
    nx,
    ny,
    nz,
):
    """Save the Q-R joint PDF histogram and normalized density to HDF5."""
    stem = _spectra_output_stem(filename)
    output_path = f"{stem}_qr_joint_pdf.h5"

    with h5py.File(output_path, "w") as hf:
        hf.attrs["source_file"] = os.path.abspath(filename)
        hf.attrs["step"] = str(step_number)
        hf.attrs["time"] = float(time_value)
        hf.attrs["grid_shape"] = np.asarray((nx, ny, nz), dtype=np.int64)
        hf.attrs["avg_SijSij"] = float(qr_result["avg_sij_sij"])
        hf.attrs["total_samples"] = int(qr_result["total_samples"])
        hf.attrs["q_min"] = float(qr_result["q_min"])
        hf.attrs["q_max"] = float(qr_result["q_max"])
        hf.attrs["r_min"] = float(qr_result["r_min"])
        hf.attrs["r_max"] = float(qr_result["r_max"])
        hf.attrs["q_normalization"] = "Q / <SijSij>"
        hf.attrs["r_normalization"] = "R / <SijSij>^(3/2)"

        hf.create_dataset("q_edges", data=np.asarray(qr_result["q_edges"], dtype=np.float64))
        hf.create_dataset("r_edges", data=np.asarray(qr_result["r_edges"], dtype=np.float64))
        hf.create_dataset("q_centers", data=np.asarray(qr_result["q_centers"], dtype=np.float64))
        hf.create_dataset("r_centers", data=np.asarray(qr_result["r_centers"], dtype=np.float64))
        hf.create_dataset("counts", data=np.asarray(qr_result["counts"], dtype=np.float64))
        hf.create_dataset("joint_pdf", data=np.asarray(qr_result["joint_pdf"], dtype=np.float64))

    print(f"Saved Q-R joint PDF data: {output_path}")
    return output_path


def plot_qr_joint_pdf(qr_result, filename):
    """Save a PDF plot of the normalized Q-R joint PDF with R on x and Q on y."""
    stem = _spectra_output_stem(filename)
    output_path = f"{stem}_qr_joint_pdf.pdf"

    q_centers = np.asarray(qr_result["q_centers"], dtype=np.float64)
    r_centers = np.asarray(qr_result["r_centers"], dtype=np.float64)
    joint_pdf = np.asarray(qr_result["joint_pdf"], dtype=np.float64)
    pdf_cap = 0.025
    contour_levels = np.linspace(0.0, pdf_cap, 21, dtype=np.float64)
    # Match the literature-style plot where all values above the top isovalue
    # saturate to the same dark red instead of stretching the color scale.
    pdf_to_plot = np.clip(joint_pdf, contour_levels[0], contour_levels[-1])
    r_curve = np.linspace(-0.8, 0.8, 800, dtype=np.float64)
    q_curve = -np.cbrt((27.0 / 4.0) * (r_curve**2))

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        filled = ax.contourf(
            r_centers,
            q_centers,
            pdf_to_plot,
            levels=contour_levels,
            cmap="RdBu_r",
            extend="max",
        )
        ax.contour(
            r_centers,
            q_centers,
            pdf_to_plot,
            levels=contour_levels[1:-1:2],
            colors="k",
            linewidths=0.4,
            alpha=0.35,
        )
        ax.plot(r_curve, q_curve, color="black", linewidth=1.5)
        colorbar = fig.colorbar(filled, ax=ax)
        colorbar.set_label(r"$\mathrm{joint\ PDF}$")
        ax.set_xlabel(r"$R / \langle S_{ij} S_{ij} \rangle^{3/2}$")
        ax.set_ylabel(r"$Q / \langle S_{ij} S_{ij} \rangle$")
        ax.set_title(r"Normalized $Q$-$R$ Joint PDF")
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    print(f"Saved Q-R joint PDF plot: {output_path}")
    return output_path
