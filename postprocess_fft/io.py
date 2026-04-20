"""Input/output helpers for FFT post-processing."""

from __future__ import annotations

import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

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


def read_structured_global_fields(filename):
    """Read full structured HDF5 velocity fields on one rank."""
    with h5py.File(filename, "r") as hf:
        vx = np.asarray(hf["fields"]["vx"][:], dtype=np.float64)
        vy = np.asarray(hf["fields"]["vy"][:], dtype=np.float64)
        vz = np.asarray(hf["fields"]["vz"][:], dtype=np.float64)
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


def save_structure_function(
    r_values,
    s_longitudinal,
    filename,
    step_number,
    time_value,
    domain_length,
    large_r_reference,
    large_r_relative_difference,
):
    """Save isotropic longitudinal structure function data beside the spectra."""
    stem = _spectra_output_stem(filename)
    output_path = f"{stem}_structure_function.txt"

    summary = np.column_stack(
        (
            np.asarray(r_values, dtype=np.float64),
            np.asarray(s_longitudinal, dtype=np.float64),
        )
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"# Step: {step_number}, Time: {float(time_value):.16e}\n")
        handle.write(f"# Domain length used in kernel: {float(domain_length):.16e}\n")
        handle.write(
            f"# Large-r reference (4/3 * sum(E_total)): "
            f"{float(large_r_reference):.16e}\n"
        )
        handle.write(
            f"# Relative difference at largest saved r: "
            f"{float(large_r_relative_difference):.16e}\n"
        )
        handle.write(f"# Columns: {'r':>23s}, {'S_L':>23s}\n")
        for row in summary:
            handle.write(", ".join(f"{value:>23.16e}" for value in row) + "\n")

    return output_path


def save_structure_functions(
    r_values,
    s2_longitudinal,
    s3_x,
    s3_y,
    s3_z,
    s3_avg,
    shell_r_values,
    s3_shell,
    shell_counts,
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
    r_sampling,
):
    """Save second- and third-order structure functions in one combined text file."""
    stem = _spectra_output_stem(filename)
    output_path = f"{stem}_structure_function.txt"

    summary = np.column_stack(
        (
            np.asarray(r_values, dtype=np.float64),
            np.asarray(s2_longitudinal, dtype=np.float64),
            np.asarray(s3_x, dtype=np.float64),
            np.asarray(s3_y, dtype=np.float64),
            np.asarray(s3_z, dtype=np.float64),
            np.asarray(s3_avg, dtype=np.float64),
        )
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"# Step: {step_number}, Time: {float(time_value):.16e}\n")
        handle.write(f"# Domain length used in kernel/physical shifts: {float(domain_length):.16e}\n")
        handle.write(f"# Axis-aligned r sampling: {r_sampling}\n")
        handle.write(
            f"# Large-r reference for S2 (4/3 * sum(E_total)): "
            f"{float(large_r_reference):.16e}\n"
        )
        handle.write(
            f"# Relative difference in S2 at reference r = {float(large_r_reference_r):.16e}: "
            f"{float(large_r_relative_difference):.16e}\n"
        )
        handle.write(
            f"# Max absolute directional spread in S3: {float(max_abs_directional_spread):.16e} "
            f"at r = {float(r_at_max_abs_directional_spread):.16e}\n"
        )
        handle.write(
            f"# Max relative directional spread in S3: {float(max_rel_directional_spread):.16e} "
            f"at r = {float(r_at_max_rel_directional_spread):.16e}\n"
        )
        handle.write(
            f"# Columns: {'r':>23s}, {'S2_L':>23s}, {'S3_x':>23s}, {'S3_y':>23s}, "
            f"{'S3_z':>23s}, {'S3_avg':>23s}\n"
        )
        handle.write("[main]\n")
        for row in summary:
            handle.write(", ".join(f"{value:>23.16e}" for value in row) + "\n")
        if shell_r_values is not None and s3_shell is not None and shell_counts is not None:
            shell_summary = np.column_stack(
                (
                    np.asarray(shell_r_values, dtype=np.float64),
                    np.asarray(s3_shell, dtype=np.float64),
                    np.asarray(shell_counts, dtype=np.float64),
                )
            )
            handle.write("[shell]\n")
            handle.write("# Shell average uses shortest-periodic lattice vectors binned onto r = m dx shells.\n")
            handle.write(
                f"# Columns: {'r':>23s}, {'S3_shell':>23s}, {'shell_count':>23s}\n"
            )
            for row in shell_summary:
                handle.write(", ".join(f"{value:>23.16e}" for value in row) + "\n")

    return output_path


def save_shell_averaged_third_order_structure_function(
    r_values,
    s3_shell,
    shell_counts,
    filename,
    step_number,
    time_value,
    domain_length,
):
    """Save radially binned lattice-shell third-order structure-function data beside the spectra."""
    stem = _spectra_output_stem(filename)
    output_path = f"{stem}_structure_function_shell_third_order.txt"

    summary = np.column_stack(
        (
            np.asarray(r_values, dtype=np.float64),
            np.asarray(s3_shell, dtype=np.float64),
            np.asarray(shell_counts, dtype=np.float64),
        )
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"# Step: {step_number}, Time: {float(time_value):.16e}\n")
        handle.write(f"# Domain length used in physical shifts: {float(domain_length):.16e}\n")
        handle.write("# Shell average uses shortest-periodic lattice vectors binned onto r = m dx shells.\n")
        handle.write(
            f"# Columns: {'r':>23s}, {'S3_shell':>23s}, {'shell_count':>23s}\n"
        )
        for row in summary:
            handle.write(", ".join(f"{value:>23.16e}" for value in row) + "\n")

    return output_path


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
        hf.attrs["total_samples"] = int(qr_result["total_samples"])
        hf.attrs["total_points"] = int(qr_result["total_points"])
        hf.attrs["retained_samples"] = int(qr_result["retained_samples"])
        hf.attrs["retained_fraction"] = float(qr_result["retained_fraction"])
        hf.attrs["max_grad_fro_sq"] = float(qr_result["max_grad_fro_sq"])
        hf.attrs["filter_fraction"] = float(qr_result["filter_fraction"])
        hf.attrs["filter_threshold"] = float(qr_result["filter_threshold"])
        hf.attrs["q_min"] = float(qr_result["q_min"])
        hf.attrs["q_max"] = float(qr_result["q_max"])
        hf.attrs["r_min"] = float(qr_result["r_min"])
        hf.attrs["r_max"] = float(qr_result["r_max"])
        hf.attrs["q_normalization"] = "Q / |grad(u)|_F^2"
        hf.attrs["r_normalization"] = "R / |grad(u)|_F^3"
        hf.attrs["filter_rule"] = "|grad(u)|_F^2 / max(|grad(u)|_F^2) >= 1e-3"

        hf.create_dataset("q_edges", data=np.asarray(qr_result["q_edges"], dtype=np.float64))
        hf.create_dataset("r_edges", data=np.asarray(qr_result["r_edges"], dtype=np.float64))
        hf.create_dataset("q_centers", data=np.asarray(qr_result["q_centers"], dtype=np.float64))
        hf.create_dataset("r_centers", data=np.asarray(qr_result["r_centers"], dtype=np.float64))
        hf.create_dataset("counts", data=np.asarray(qr_result["counts"], dtype=np.float64))
        hf.create_dataset("joint_pdf", data=np.asarray(qr_result["joint_pdf"], dtype=np.float64))

    print(f"Saved Q-R joint PDF data: {output_path}")
    print(
        "  Q-R PDF normalization: q_A = Q / |grad(u)|_F^2, "
        "r_A = R / |grad(u)|_F^3"
    )
    print(
        "  Retained samples after Frobenius-norm filter: "
        f"{qr_result['retained_samples']} / {qr_result['total_points']} "
        f"({100.0 * qr_result['retained_fraction']:.2f}%)"
    )
    return output_path


def _qr_plot_settings():
    """Return the shared styling and range settings for Q-R PDF plots."""
    return {
        "r_plot_limit": 0.2,
        "q_plot_limit": 0.5,
        "colorbar_vmin": 1.0e-3,
        "colorbar_vmax": 1.0e2,
        "cmap": "RdBu_r",
        "line_contour_probabilities": (0.05, 0.15, 0.25, 0.50),
    }


def _qr_enclosed_probability_contour_level(joint_pdf, q_edges, r_edges, enclosed_probability=0.9):
    """Return the PDF isovalue enclosing the highest-density region with the target mass."""
    bin_area = np.outer(np.diff(q_edges), np.diff(r_edges))
    cell_probability = np.asarray(joint_pdf, dtype=np.float64) * bin_area
    flat_pdf = np.asarray(joint_pdf, dtype=np.float64).ravel(order="C")
    flat_probability = cell_probability.ravel(order="C")
    positive_mask = flat_pdf > 0.0
    if not np.any(positive_mask):
        return None

    order = np.argsort(flat_pdf[positive_mask])[::-1]
    sorted_pdf = flat_pdf[positive_mask][order]
    sorted_probability = flat_probability[positive_mask][order]
    cumulative_probability = np.cumsum(sorted_probability)
    contour_index = int(np.searchsorted(cumulative_probability, enclosed_probability, side="left"))
    contour_index = min(contour_index, sorted_pdf.size - 1)
    return float(sorted_pdf[contour_index])


def _qr_enclosed_probability_contour_levels(joint_pdf, q_edges, r_edges, enclosed_probabilities):
    """Return contour levels for a sequence of enclosed-probability targets."""
    levels = {}
    for probability in enclosed_probabilities:
        level = _qr_enclosed_probability_contour_level(
            joint_pdf,
            q_edges,
            r_edges,
            enclosed_probability=probability,
        )
        if level is not None:
            levels[float(probability)] = float(level)
    return levels


def _print_qr_plot_summary(output_path, backend_name, settings, line_contour_levels, contour_90_level):
    """Print a consistent summary of the Q-R plot that was written."""
    print(f"Saved Q-R joint PDF plot: {output_path}")
    print(f"  Plot backend: {backend_name}")
    print(
        "  Plot bounds: "
        f"r_A in [{-settings['r_plot_limit']:.1f}, {settings['r_plot_limit']:.1f}], "
        f"q_A in [{-settings['q_plot_limit']:.1f}, {settings['q_plot_limit']:.1f}]"
    )
    print(
        "  Colorbar scale: "
        f"log10 joint PDF from {settings['colorbar_vmin']:.0e} to {settings['colorbar_vmax']:.0e}"
    )
    if line_contour_levels:
        formatted = ", ".join(
            f"{int(round(probability * 100.0))}% -> {level:.6e}"
            for probability, level in sorted(line_contour_levels.items())
        )
        print(f"  Enclosed-probability contour levels: {formatted}")
    if contour_90_level is not None:
        print(f"  90% probability contour level: {contour_90_level:.6e}")


def _plot_qr_joint_pdf_matplotlib(qr_result, output_path):
    """Render the Q-R PDF using the existing Matplotlib contour workflow."""
    settings = _qr_plot_settings()
    q_centers = np.asarray(qr_result["q_centers"], dtype=np.float64)
    r_centers = np.asarray(qr_result["r_centers"], dtype=np.float64)
    joint_pdf = np.asarray(qr_result["joint_pdf"], dtype=np.float64)
    q_edges = np.asarray(qr_result["q_edges"], dtype=np.float64)
    r_edges = np.asarray(qr_result["r_edges"], dtype=np.float64)
    # Plot the positive PDF on logarithmic contour levels to resolve both the
    # dense core and the weaker tails in the joint distribution.
    contour_levels = np.logspace(
        np.log10(settings["colorbar_vmin"]),
        np.log10(settings["colorbar_vmax"]),
        21,
        dtype=np.float64,
    )
    line_contour_levels = _qr_enclosed_probability_contour_levels(
        joint_pdf,
        q_edges,
        r_edges,
        settings["line_contour_probabilities"],
    )
    contour_90_level = _qr_enclosed_probability_contour_level(joint_pdf, q_edges, r_edges)
    pdf_to_plot = np.ma.masked_less_equal(joint_pdf, 0.0)
    r_curve = np.linspace(-settings["r_plot_limit"], settings["r_plot_limit"], 800, dtype=np.float64)
    q_curve = -np.cbrt((27.0 / 4.0) * (r_curve**2))

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        filled = ax.contourf(
            r_centers,
            q_centers,
            pdf_to_plot,
            levels=contour_levels,
            cmap=settings["cmap"],
            norm=LogNorm(vmin=settings["colorbar_vmin"], vmax=settings["colorbar_vmax"]),
            extend="both",
        )
        if line_contour_levels:
            ax.contour(
                r_centers,
                q_centers,
                pdf_to_plot,
                levels=np.asarray(sorted(line_contour_levels.values()), dtype=np.float64),
                colors="k",
                linewidths=0.9,
                alpha=0.8,
            )
        ax.plot(r_curve, q_curve, color="black", linewidth=1.5)
        if contour_90_level is not None:
            # The magenta contour marks the highest-density region containing
            # 90% of the total probability mass in the joint PDF.
            ax.contour(
                r_centers,
                q_centers,
                joint_pdf,
                levels=[contour_90_level],
                colors=["magenta"],
                linewidths=1.6,
            )
        colorbar = fig.colorbar(filled, ax=ax)
        colorbar.set_label(r"$\mathrm{joint\ PDF}$")
        ax.set_xlabel(r"$r_A$")
        ax.set_ylabel(r"$q_A$")
        ax.set_title(r"Frobenius-Normalized $Q$-$R$ Joint PDF")
        ax.set_xlim(-settings["r_plot_limit"], settings["r_plot_limit"])
        ax.set_ylim(-settings["q_plot_limit"], settings["q_plot_limit"])
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    _print_qr_plot_summary(output_path, "matplotlib", settings, line_contour_levels, contour_90_level)
    return output_path


def _plot_qr_joint_pdf_yt(qr_result, output_path):
    """Render the Q-R PDF with yt.PhasePlot and Matplotlib overlays."""
    import yt
    from yt.visualization.profile_plotter import PhasePlot

    settings = _qr_plot_settings()
    q_edges = np.asarray(qr_result["q_edges"], dtype=np.float64)
    r_edges = np.asarray(qr_result["r_edges"], dtype=np.float64)
    q_centers = np.asarray(qr_result["q_centers"], dtype=np.float64)
    r_centers = np.asarray(qr_result["r_centers"], dtype=np.float64)
    joint_pdf = np.asarray(qr_result["joint_pdf"], dtype=np.float64)
    line_contour_levels = _qr_enclosed_probability_contour_levels(
        joint_pdf,
        q_edges,
        r_edges,
        settings["line_contour_probabilities"],
    )
    contour_90_level = _qr_enclosed_probability_contour_level(joint_pdf, q_edges, r_edges)

    rr, qq = np.meshgrid(r_centers, q_centers)
    particle_count = rr.size
    particle_data = {
        "particle_position_x": np.zeros(particle_count, dtype=np.float64),
        "particle_position_y": np.zeros(particle_count, dtype=np.float64),
        "particle_position_z": np.zeros(particle_count, dtype=np.float64),
        # One synthetic particle per (r_A, q_A) bin center lets yt render the
        # already-computed joint PDF through its PhasePlot/profile machinery.
        "r_bin": rr.ravel(order="C"),
        "q_bin": qq.ravel(order="C"),
        "joint_pdf_value": joint_pdf.ravel(order="C"),
    }

    ds = yt.load_particles(particle_data, length_unit=1.0)
    ad = ds.all_data()
    profile = yt.create_profile(
        ad,
        [("io", "r_bin"), ("io", "q_bin")],
        [("io", "joint_pdf_value")],
        n_bins=(len(r_centers), len(q_centers)),
        extrema={
            ("io", "r_bin"): (float(r_edges[0]), float(r_edges[-1])),
            ("io", "q_bin"): (float(q_edges[0]), float(q_edges[-1])),
        },
        weight_field=None,
        logs={
            ("io", "r_bin"): False,
            ("io", "q_bin"): False,
        },
    )
    phase = PhasePlot.from_profile(profile, figure_size=7.0, fontsize=16)
    phase.set_cmap(("io", "joint_pdf_value"), settings["cmap"])
    phase.set_zlim(("io", "joint_pdf_value"), settings["colorbar_vmin"], settings["colorbar_vmax"])
    phase.render()

    plot = phase.plots[("io", "joint_pdf_value")]
    ax = plot.axes
    ax.set_xlim(-settings["r_plot_limit"], settings["r_plot_limit"])
    ax.set_ylim(-settings["q_plot_limit"], settings["q_plot_limit"])
    ax.set_xlabel(r"$r_A$")
    ax.set_ylabel(r"$q_A$")
    ax.set_title(r"Frobenius-Normalized $Q$-$R$ Joint PDF")
    ax.grid(True, which="both", alpha=0.2)

    r_curve = np.linspace(-settings["r_plot_limit"], settings["r_plot_limit"], 800, dtype=np.float64)
    q_curve = -np.cbrt((27.0 / 4.0) * (r_curve**2))
    ax.plot(r_curve, q_curve, color="black", linewidth=1.5)
    if line_contour_levels:
        ax.contour(
            r_centers,
            q_centers,
            np.ma.masked_less_equal(joint_pdf, 0.0),
            levels=np.asarray(sorted(line_contour_levels.values()), dtype=np.float64),
            colors="k",
            linewidths=0.9,
            alpha=0.8,
        )
    if contour_90_level is not None:
        ax.contour(
            r_centers,
            q_centers,
            joint_pdf,
            levels=[contour_90_level],
            colors=["magenta"],
            linewidths=1.6,
        )

    if hasattr(plot, "cb"):
        plot.cb.set_label(r"$\mathrm{joint\ PDF}$")

    plot.figure.savefig(output_path, bbox_inches="tight")
    plt.close(plot.figure)

    _print_qr_plot_summary(output_path, "yt", settings, line_contour_levels, contour_90_level)
    return output_path


def plot_qr_joint_pdf(qr_result, filename):
    """Save a PDF plot of the Frobenius-normalized Q-R joint PDF with yt when available."""
    stem = _spectra_output_stem(filename)
    output_path = f"{stem}_qr_joint_pdf.pdf"
    backend = os.environ.get("TURB_POSTPROCESS_QR_PLOT_BACKEND", "yt").strip().lower()

    if backend not in {"yt", "matplotlib"}:
        raise ValueError(
            "Unsupported Q-R plot backend "
            f"{backend!r}. Use one of: yt, matplotlib."
        )

    if backend == "yt":
        try:
            return _plot_qr_joint_pdf_yt(qr_result, output_path)
        except Exception as exc:
            print(f"yt PhasePlot backend unavailable, falling back to Matplotlib: {exc}")

    return _plot_qr_joint_pdf_matplotlib(qr_result, output_path)
