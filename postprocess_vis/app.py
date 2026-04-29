"""Scalable slice-plot workflow for structured velocity HDF5 files."""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
from mpi4py import MPI

from postprocess_fft.analysis_context import DistributedAnalysisContext
from postprocess_fft.io import read_structured_local_fields
from postprocess_fft.layout import box_shape
from postprocess_fft.layout import box_slices
from postprocess_fft.layout import build_boxes
from postprocess_fft.layout import choose_proc_grid
from postprocess_fft.transform import backward_field
from postprocess_fft.transform import forward_field
from postprocess_fft.transform import get_backend
from postprocess_fft.transform import local_wavenumber_mesh
from postprocess_fft.common import global_mean_energy
from postprocess_fft.common import global_range
from postprocess_fft.common import heffte
from postprocess_lib.prepare import ensure_structured_h5
from postprocess_vis.field_specs import BUILTIN_FIELD_MAP
from postprocess_vis.field_specs import DENSITY_DATASET_NAME
from postprocess_vis.field_specs import DERIVED_DATASET_NAMES
from postprocess_vis.field_specs import DERIVED_FIELD_FAMILIES
from postprocess_vis.field_specs import MACH_NUMBER_FIELD_NAME
from postprocess_vis.field_specs import PRESSURE_DATASET_NAME
from postprocess_vis.field_specs import SOUND_SPEED_FIELD_NAME
from postprocess_vis.field_specs import build_available_field_specs
from postprocess_vis.field_specs import finalize_requested_field_names
from postprocess_vis.normalization_labels import format_plot_label
from postprocess_vis.pdfs import DEFAULT_FIELD_PDF_BINS
from postprocess_vis.pdfs import compute_distributed_field_pdf
from postprocess_vis.pdfs import default_field_pdf_specs
from postprocess_vis.pdfs import field_pdf_output_path
from postprocess_vis.pdfs import plot_field_pdf
from postprocess_vis.pdfs import plot_smoothed_field_pdf
from postprocess_vis.pdfs import print_field_pdf_summary
from postprocess_vis.pdfs import smooth_field_pdf_for_plot
from postprocess_vis.pdfs import write_field_pdf_metadata
from postprocess_vis.slice_data import default_slice_data_output
from postprocess_vis.slice_data import initialize_slice_data_file
from postprocess_vis.slice_data import save_pdf_serial
from postprocess_vis.slice_data import storage_slice_tag
from postprocess_vis.slice_data import write_slice_stats_serial
from postprocess_vis.slice_data import write_slice_plane_parallel
from postprocess_vis.slice_data import write_slice_plane_serial

PLANE_NAMES = {
    "x": "yz",
    "y": "zx",
    "z": "xy",
}

PLANE_AXES = {
    "x": ("z", "y"),
    "y": ("z", "x"),
    "z": ("y", "x"),
}

NEAR_ZERO_PDF_STD_TOL = 1.0e-12


def log_rank0(rank, message):
    """Print one progress message from rank 0 and flush immediately."""
    if rank == 0:
        print(message, flush=True)


def _requested_derived_field_labels(
    *,
    needs_vorticity,
    needs_divergence,
    needs_sound_speed,
    needs_mach_number,
    needs_qcriterion,
    needs_rcriterion,
    needs_density_gradient,
    needs_velocity_gradient_u12,
):
    """Return user-facing labels for the derived fields requested in this run."""
    labels = []
    if needs_vorticity:
        labels.append("vorticity")
    if needs_divergence:
        labels.append("velocity-divergence")
    if needs_sound_speed:
        labels.append("sound-speed")
    if needs_mach_number:
        labels.append("Mach-number")
    if needs_qcriterion:
        labels.append("Q-criterion")
    if needs_rcriterion:
        labels.append("R-criterion")
    if needs_density_gradient:
        labels.append("density-gradient magnitude")
    if needs_velocity_gradient_u12:
        labels.append("velocity-gradient u_{1,2}")
    return labels


def split_axis(length, parts):
    """Split a 1D index range into nearly equal contiguous chunks."""
    base, remainder = divmod(length, parts)
    start = 0
    chunks = []
    for i in range(parts):
        stop = start + base + (1 if i < remainder else 0)
        chunks.append((start, stop))
        start = stop
    return chunks


def canonical_field_name(field_name):
    """Return the HDF5 field name and human-readable label."""
    if field_name not in BUILTIN_FIELD_MAP:
        raise ValueError(f"Unsupported field '{field_name}'.")
    return BUILTIN_FIELD_MAP[field_name]


def available_field_specs(filepath):
    """Return the built-in and dataset-backed field specs available in one HDF5 file."""
    with h5py.File(filepath, "r") as hf:
        return build_available_field_specs(hf["fields"])


def _yt_font_style():
    return {
        "family": "serif",
        "size": 20,
    }


def read_grid_metadata(filepath):
    """Read grid coordinates and metadata from one structured HDF5 file."""
    with h5py.File(filepath, "r") as hf:
        x_full = np.asarray(hf["grid"]["x"][:], dtype=np.float64)
        y_full = np.asarray(hf["grid"]["y"][:], dtype=np.float64)
        z_full = np.asarray(hf["grid"]["z"][:], dtype=np.float64)
        periodic_duplicate_last = bool(hf.attrs.get("periodic_duplicate_last", True))
        step = str(hf.attrs.get("step", "unknown"))
        time_val = float(hf.attrs.get("time", 0.0))

    if periodic_duplicate_last and len(x_full) > 1 and len(y_full) > 1 and len(z_full) > 1:
        x_coords = x_full[:-1]
        y_coords = y_full[:-1]
        z_coords = z_full[:-1]
    else:
        x_coords = x_full
        y_coords = y_full
        z_coords = z_full

    return {
        "x": x_coords,
        "y": y_coords,
        "z": z_coords,
        "shape": (len(x_coords), len(y_coords), len(z_coords)),
        "dx": float(x_full[1] - x_full[0]) if len(x_full) > 1 else 1.0,
        "dy": float(y_full[1] - y_full[0]) if len(y_full) > 1 else 1.0,
        "dz": float(z_full[1] - z_full[0]) if len(z_full) > 1 else 1.0,
        "step": step,
        "time": time_val,
    }


def open_h5_for_parallel_read(filepath, comm):
    """Use MPI-enabled HDF5 reads when available."""
    if h5py.get_config().mpi:
        return h5py.File(filepath, "r", driver="mpio", comm=comm)
    return h5py.File(filepath, "r")


def read_structured_local_dataset(filepath, dataset_name, local_box, comm):
    """Read one rank-local dataset block from the structured HDF5 file."""
    sx, sy, sz = box_slices(local_box)
    with open_h5_for_parallel_read(filepath, comm) as hf:
        return np.asarray(hf["fields"][dataset_name][sx, sy, sz], dtype=np.float64)


def parse_slice_spec(spec):
    """Parse one slice selector of the form axis:selector."""
    if ":" not in spec:
        raise ValueError(
            f"Invalid slice spec '{spec}'. Expected axis:selector, "
            "for example z:center, y:idx=10, x:frac=0.25, z:coord=0.5."
        )

    axis, selector = spec.split(":", 1)
    axis = axis.strip().lower()
    selector = selector.strip().lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Invalid slice axis '{axis}'.")

    if selector == "center":
        return axis, ("center", None)
    if selector.startswith("idx="):
        return axis, ("idx", int(selector.split("=", 1)[1]))
    if selector.startswith("frac="):
        return axis, ("frac", float(selector.split("=", 1)[1]))
    if selector.startswith("coord="):
        return axis, ("coord", float(selector.split("=", 1)[1]))

    raise ValueError(
        f"Unsupported slice selector '{selector}'. "
        "Use center, idx=<int>, frac=<0..1>, or coord=<float>."
    )


def resolve_slice_index(coords, selector):
    """Convert a parsed selector into a concrete plane index."""
    mode, value = selector
    npts = len(coords)
    if npts == 0:
        raise ValueError("Cannot select a slice from an empty axis.")

    if mode == "center":
        return npts // 2, "center"

    if mode == "idx":
        index = int(value)
        if index < 0 or index >= npts:
            raise ValueError(f"Slice index {index} is out of bounds for axis length {npts}.")
        return index, f"idx{index:05d}"

    if mode == "frac":
        frac = float(value)
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"Slice fraction {frac} is outside [0, 1].")
        index = int(np.rint(frac * (npts - 1)))
        return index, f"frac{frac:.6f}".replace(".", "p")

    if mode == "coord":
        coord = float(value)
        index = int(np.argmin(np.abs(coords - coord)))
        return index, f"coord{coord:.6f}".replace(".", "p")

    raise ValueError(f"Unknown selector mode '{mode}'.")


def gather_plane(axis, local_bounds, local_plane, shape, comm):
    """Gather one distributed 2D plane back to rank 0."""
    rank = comm.Get_rank()
    gathered = comm.gather((local_bounds, local_plane), root=0)
    if rank != 0:
        return None

    nx, ny, nz = shape
    if axis == "x":
        plane = np.zeros((ny, nz), dtype=np.float64)
        for bounds, piece in gathered:
            if piece is None:
                continue
            y0, y1, z0, z1 = bounds
            plane[y0:y1, z0:z1] = piece
        return plane

    if axis == "y":
        plane = np.zeros((nx, nz), dtype=np.float64)
        for bounds, piece in gathered:
            if piece is None:
                continue
            x0, x1, z0, z1 = bounds
            plane[x0:x1, z0:z1] = piece
        return plane

    plane = np.zeros((nx, ny), dtype=np.float64)
    for bounds, piece in gathered:
        if piece is None:
            continue
        x0, x1, y0, y1 = bounds
        plane[x0:x1, y0:y1] = piece
    return plane


def extract_plane_from_boxes_local(axis, local_box, local_field, plane_index):
    """Extract one rank-local piece of a plane from a generic 3D box decomposition."""
    sx, sy, sz = box_slices(local_box)

    bounds = None
    piece = None
    if axis == "x":
        if sx.start <= plane_index < sx.stop:
            local_index = plane_index - sx.start
            piece = np.asarray(local_field[local_index, :, :], dtype=np.float64)
            bounds = (sy.start, sy.stop, sz.start, sz.stop)
    elif axis == "y":
        if sy.start <= plane_index < sy.stop:
            local_index = plane_index - sy.start
            piece = np.asarray(local_field[:, local_index, :], dtype=np.float64)
            bounds = (sx.start, sx.stop, sz.start, sz.stop)
    else:
        if sz.start <= plane_index < sz.stop:
            local_index = plane_index - sz.start
            piece = np.asarray(local_field[:, :, local_index], dtype=np.float64)
            bounds = (sx.start, sx.stop, sy.start, sy.stop)

    return bounds, piece


def gather_plane_from_boxes(axis, local_box, local_field, shape, plane_index, comm):
    """Gather one plane from a generic 3D box decomposition."""
    local_bounds, local_plane = extract_plane_from_boxes_local(axis, local_box, local_field, plane_index)
    return gather_plane(axis, local_bounds, local_plane, shape, comm)


def extract_plane_parallel_local(filepath, dataset_name, axis, plane_index, meta, comm):
    """Read the rank-local piece of one requested HDF5 plane in parallel."""
    shape = meta["shape"]
    nx, ny, nz = shape
    x_ranges = split_axis(nx, comm.Get_size())
    x_start, x_stop = x_ranges[comm.Get_rank()]

    local_bounds = None
    local_plane = None
    with open_h5_for_parallel_read(filepath, comm) as hf:
        if dataset_name == "velocity_magnitude":
            vx_field = hf["fields"]["vx"]
            vy_field = hf["fields"]["vy"]
            vz_field = hf["fields"]["vz"]
            if axis == "x":
                if x_start <= plane_index < x_stop:
                    vx_plane = np.asarray(vx_field[plane_index, :ny, :nz], dtype=np.float64)
                    vy_plane = np.asarray(vy_field[plane_index, :ny, :nz], dtype=np.float64)
                    vz_plane = np.asarray(vz_field[plane_index, :ny, :nz], dtype=np.float64)
                    local_plane = np.sqrt(vx_plane**2 + vy_plane**2 + vz_plane**2)
                    local_bounds = (0, ny, 0, nz)
            elif axis == "y":
                vx_plane = np.asarray(vx_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                vy_plane = np.asarray(vy_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                vz_plane = np.asarray(vz_field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                local_plane = np.sqrt(vx_plane**2 + vy_plane**2 + vz_plane**2)
                local_bounds = (x_start, x_stop, 0, nz)
            else:
                vx_plane = np.asarray(vx_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                vy_plane = np.asarray(vy_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                vz_plane = np.asarray(vz_field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                local_plane = np.sqrt(vx_plane**2 + vy_plane**2 + vz_plane**2)
                local_bounds = (x_start, x_stop, 0, ny)
        else:
            field = hf["fields"][dataset_name]
            if axis == "x":
                if x_start <= plane_index < x_stop:
                    local_plane = np.asarray(field[plane_index, :ny, :nz], dtype=np.float64)
                    local_bounds = (0, ny, 0, nz)
            elif axis == "y":
                local_plane = np.asarray(field[x_start:x_stop, plane_index, :nz], dtype=np.float64)
                local_bounds = (x_start, x_stop, 0, nz)
            else:
                local_plane = np.asarray(field[x_start:x_stop, :ny, plane_index], dtype=np.float64)
                local_bounds = (x_start, x_stop, 0, ny)

    return local_bounds, local_plane


def extract_plane_parallel(filepath, dataset_name, axis, plane_index, meta, comm):
    """Read only the requested HDF5 plane in parallel."""
    local_bounds, local_plane = extract_plane_parallel_local(filepath, dataset_name, axis, plane_index, meta, comm)
    return gather_plane(axis, local_bounds, local_plane, meta["shape"], comm)


def compute_local_derived_fields(
    filepath,
    meta,
    comm,
    backend_name,
    include_vorticity=False,
    include_divergence=False,
    include_qcriterion=False,
    include_rcriterion=False,
    include_sound_speed=False,
    include_mach_number=False,
    include_density_gradient=False,
    include_velocity_gradient_u12=False,
    density_dataset_name=None,
    pressure_dataset_name=None,
    thermo_gamma=1.4,
    analysis_context=None,
):
    """Compute distributed real-space derived velocity fields with HeFFTe."""
    if analysis_context is None:
        analysis_context = DistributedAnalysisContext.from_structured_h5(filepath, meta, comm, backend_name)

    derived_fields = {
        "local_box": analysis_context.local_box,
    }

    needs_thermo_fields = include_sound_speed or include_mach_number

    if include_vorticity:
        omega_x, omega_y, omega_z = analysis_context.get_vorticity_components()
        derived_fields["omega_x"] = omega_x
        derived_fields["omega_y"] = omega_y
        derived_fields["omega_z"] = omega_z
        derived_fields["vorticity_magnitude"] = analysis_context.get_vorticity_magnitude()

    if include_divergence or include_qcriterion or include_rcriterion:
        div_u = analysis_context.get_divergence()
        if include_divergence:
            derived_fields["div_u"] = div_u
        if include_qcriterion:
            derived_fields["q_criterion"] = analysis_context.get_qcriterion()

        if include_rcriterion:
            derived_fields["r_criterion"] = analysis_context.get_rcriterion()

    if include_density_gradient:
        if not density_dataset_name:
            raise ValueError("A density dataset name is required to compute the density-gradient magnitude.")
        derived_fields["density_gradient_magnitude"] = analysis_context.get_density_gradient_magnitude(
            density_dataset_name
        )

    if include_velocity_gradient_u12:
        derived_fields["dux_dy"] = analysis_context.get_velocity_gradients()["dux_dy"]

    if needs_thermo_fields:
        if not density_dataset_name or not pressure_dataset_name:
            raise ValueError("Density and pressure dataset names are required to compute thermodynamic derived fields.")
        local_sound_speed = analysis_context.get_sound_speed(
            density_dataset_name,
            pressure_dataset_name,
            thermo_gamma,
        )
        derived_fields[SOUND_SPEED_FIELD_NAME] = local_sound_speed

        if include_mach_number:
            derived_fields[MACH_NUMBER_FIELD_NAME] = analysis_context.get_mach_number(
                density_dataset_name,
                pressure_dataset_name,
                thermo_gamma,
            )

    return derived_fields


def global_rms(values, comm):
    """Return the global RMS of a distributed field."""
    local_values = np.asarray(values, dtype=np.float64)
    local_sum_sq = float(np.sum(local_values**2, dtype=np.float64))
    local_count = int(local_values.size)
    global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    if global_count <= 0:
        return 0.0
    return float(np.sqrt(global_sum_sq / float(global_count)))


def global_mean(values, comm):
    """Return the global arithmetic mean of a distributed field."""
    local_values = np.asarray(values, dtype=np.float64)
    local_sum = float(np.sum(local_values, dtype=np.float64))
    local_count = int(local_values.size)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    if global_count <= 0:
        return 0.0
    return float(global_sum / float(global_count))


def global_std(values, comm):
    """Return the global population standard deviation of a distributed field."""
    local_values = np.asarray(values, dtype=np.float64)
    local_sum = float(np.sum(local_values, dtype=np.float64))
    local_sum_sq = float(np.sum(local_values**2, dtype=np.float64))
    local_count = int(local_values.size)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    if global_count <= 0:
        return 0.0
    mean = global_sum / float(global_count)
    variance = max(global_sum_sq / float(global_count) - mean * mean, 0.0)
    return float(np.sqrt(variance))


def compute_global_field_stats(filepath, dataset_name, field_family, meta, comm, derived_cache=None, analysis_context=None):
    """Return the full 3D min/max, RMS, mean, and standard deviation for one field."""
    if analysis_context is not None and dataset_name == "velocity_magnitude":
        local_values = np.sqrt(
            analysis_context.local_vx**2 + analysis_context.local_vy**2 + analysis_context.local_vz**2
        )
    elif analysis_context is not None and dataset_name == "vx":
        local_values = np.asarray(analysis_context.local_vx, dtype=np.float64)
    elif analysis_context is not None and dataset_name == "vy":
        local_values = np.asarray(analysis_context.local_vy, dtype=np.float64)
    elif analysis_context is not None and dataset_name == "vz":
        local_values = np.asarray(analysis_context.local_vz, dtype=np.float64)
    elif dataset_name in DERIVED_DATASET_NAMES:
        if derived_cache is None or dataset_name not in derived_cache:
            raise ValueError(f"Derived cache is missing required field '{dataset_name}'.")
        local_values = np.asarray(derived_cache[dataset_name], dtype=np.float64)
    elif analysis_context is not None:
        local_values = np.asarray(analysis_context.get_local_dataset(dataset_name), dtype=np.float64)
    else:
        shape = meta["shape"]
        proc_grid = choose_proc_grid(shape, comm.Get_size())
        boxes = build_boxes(shape, proc_grid)
        local_box = boxes[comm.Get_rank()]
        if dataset_name == "velocity_magnitude":
            local_vx, local_vy, local_vz = read_structured_local_fields(filepath, local_box, comm)
            local_values = np.sqrt(local_vx**2 + local_vy**2 + local_vz**2)
        else:
            local_values = read_structured_local_dataset(filepath, dataset_name, local_box, comm)

    global_min, global_max = global_range(local_values, comm)
    return {
        "global_min": float(global_min),
        "global_max": float(global_max),
        "global_rms": global_rms(local_values, comm),
        "global_mean": global_mean(local_values, comm),
        "global_std": global_std(local_values, comm),
    }


def print_global_field_stats_block(title, stats):
    """Print one formatted global-statistics block on rank 0."""
    print(f"{title}:", flush=True)
    print(f"  min={stats['global_min']:.6g}", flush=True)
    print(f"  max={stats['global_max']:.6g}", flush=True)
    print(f"  rms={stats['global_rms']:.6g}", flush=True)
    print(f"  std={stats['global_std']:.6g}", flush=True)
    print(f"  avg={stats['global_mean']:.6g}", flush=True)


def output_stem(source_path, field_label):
    """Build a clean default filename stem with the plotted field name once in front."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    marker = "_sampled_data"
    if marker in base:
        _, suffix = base.split(marker, 1)
        return f"{field_label}{marker}{suffix}"
    if base == field_label or base.startswith(f"{field_label}_"):
        return base
    if base.endswith(f"_{field_label}"):
        return base[: -(len(field_label) + 1)]
    return f"{field_label}_{base}"


def output_name(data_file, field_label, axis, slice_tag, output_format):
    """Return the default output name for one slice image."""
    directory = os.path.dirname(os.path.abspath(data_file))
    output_dir = os.path.join(directory, "slice_plots")
    os.makedirs(output_dir, exist_ok=True)
    base = output_stem(data_file, field_label)
    if slice_tag in {"xy_center", "xy_face", "yz_face", "zx_face"}:
        return os.path.join(output_dir, f"{base}_{slice_tag}.{output_format}")
    return os.path.join(output_dir, f"{base}_{PLANE_NAMES[axis]}_{slice_tag}.{output_format}")


def compute_and_store_full_field_pdfs(
    prepared_path,
    pdf_specs,
    stats_cache,
    derived_cache,
    meta,
    comm,
    *,
    save_slice_data,
    slice_data_path,
    output_format,
    plot,
    pdf_bins,
    analysis_context=None,
):
    """Compute configured full-field PDFs, store them in HDF5, and plot them on rank 0."""
    rank = comm.Get_rank()
    outputs = []
    if not pdf_specs:
        return outputs

    if not save_slice_data:
        log_rank0(rank, "Skipping full-field PDF generation because slice-data output is disabled.")
        return outputs

    field_lookup = available_field_specs(prepared_path)
    if analysis_context is not None:
        local_box = analysis_context.local_box
    else:
        proc_grid = choose_proc_grid(meta["shape"], comm.Get_size())
        boxes = build_boxes(meta["shape"], proc_grid)
        local_box = boxes[rank]

    for pdf_spec in pdf_specs:
        source_field = str(pdf_spec["source_field"])
        if source_field not in field_lookup:
            raise ValueError(
                f"Cannot compute PDF '{pdf_spec['pdf_name']}' because field '{source_field}' is unavailable."
            )
        dataset_name, field_label, latex_label, field_family = field_lookup[source_field]
        global_stats = stats_cache.get(dataset_name)
        if global_stats is None:
            global_stats = compute_global_field_stats(
                prepared_path,
                dataset_name,
                field_family,
                meta,
                comm,
                derived_cache=derived_cache,
                analysis_context=analysis_context,
            )
            stats_cache[dataset_name] = global_stats

        normalization = str(pdf_spec.get("normalization", "none")).strip().lower()
        normalization_scale = 1.0
        normalization_offset = 0.0
        normalization_reference_field = ""
        normalization_reference_label = ""
        normalization_reference_std = None
        normalization_reference_rms = None
        normalization_scale_field_name = source_field
        if normalization == "global_std":
            normalization_scale = float(global_stats["global_std"])
            normalization_offset = float(global_stats["global_mean"])
        elif normalization == "global_rms":
            normalization_scale = float(global_stats["global_rms"])
        elif normalization == "reference_global_rms":
            reference_field = str(pdf_spec.get("normalization_reference_field", "vorticity_magnitude"))
            if reference_field not in field_lookup:
                raise ValueError(
                    f"Cannot compute PDF '{pdf_spec['pdf_name']}' because normalization reference field "
                    f"'{reference_field}' is unavailable."
                )
            reference_dataset_name, _, reference_latex_label, reference_field_family = field_lookup[reference_field]
            reference_stats = stats_cache.get(reference_dataset_name)
            if reference_stats is None:
                reference_stats = compute_global_field_stats(
                    prepared_path,
                    reference_dataset_name,
                    reference_field_family,
                    meta,
                    comm,
                    derived_cache=derived_cache,
                    analysis_context=analysis_context,
                )
                stats_cache[reference_dataset_name] = reference_stats
            normalization_scale = float(reference_stats["global_rms"])
            normalization_reference_field = reference_field
            normalization_reference_label = reference_latex_label
            normalization_reference_std = float(reference_stats["global_std"])
            normalization_reference_rms = float(reference_stats["global_rms"])
            normalization_scale_field_name = reference_field
        elif normalization == "source_mean_vorticity_std":
            reference_field = str(pdf_spec.get("normalization_reference_field", "vorticity_magnitude"))
            if reference_field not in field_lookup:
                raise ValueError(
                    f"Cannot compute PDF '{pdf_spec['pdf_name']}' because normalization reference field "
                    f"'{reference_field}' is unavailable."
                )
            reference_dataset_name, _, reference_latex_label, reference_field_family = field_lookup[reference_field]
            reference_stats = stats_cache.get(reference_dataset_name)
            if reference_stats is None:
                reference_stats = compute_global_field_stats(
                    prepared_path,
                    reference_dataset_name,
                    reference_field_family,
                    meta,
                    comm,
                    derived_cache=derived_cache,
                    analysis_context=analysis_context,
                )
                stats_cache[reference_dataset_name] = reference_stats
            normalization_scale = float(reference_stats["global_std"])
            normalization_offset = float(global_stats["global_mean"])
            normalization_reference_field = reference_field
            normalization_reference_label = reference_latex_label
            normalization_reference_std = float(reference_stats["global_std"])
            normalization_reference_rms = float(reference_stats["global_rms"])
            normalization_scale_field_name = reference_field
        elif normalization != "none":
            raise ValueError(
                f"Unsupported PDF normalization '{normalization}' for '{pdf_spec['pdf_name']}'."
            )

        if source_field in DERIVED_DATASET_NAMES:
            if derived_cache is None or source_field not in derived_cache:
                raise ValueError(
                    f"Cannot compute PDF '{pdf_spec['pdf_name']}' because derived field '{source_field}' is unavailable."
                )
            pdf_local_values = derived_cache[source_field]
        elif analysis_context is not None and dataset_name == "velocity_magnitude":
            pdf_local_values = np.sqrt(
                analysis_context.local_vx**2 + analysis_context.local_vy**2 + analysis_context.local_vz**2
            )
        elif analysis_context is not None and dataset_name == "vx":
            pdf_local_values = analysis_context.local_vx
        elif analysis_context is not None and dataset_name == "vy":
            pdf_local_values = analysis_context.local_vy
        elif analysis_context is not None and dataset_name == "vz":
            pdf_local_values = analysis_context.local_vz
        elif dataset_name == "velocity_magnitude":
            local_vx, local_vy, local_vz = read_structured_local_fields(prepared_path, local_box, comm)
            pdf_local_values = np.sqrt(local_vx**2 + local_vy**2 + local_vz**2)
        elif analysis_context is not None:
            pdf_local_values = analysis_context.get_local_dataset(dataset_name)
        else:
            pdf_local_values = read_structured_local_dataset(prepared_path, dataset_name, local_box, comm)
        pdf_value_range = None
        near_zero_treated_as_zero = False
        if (
            normalization in {
                "global_std",
                "global_rms",
                "reference_global_rms",
                "source_mean_vorticity_std",
            }
            and normalization_scale <= NEAR_ZERO_PDF_STD_TOL
        ):
            near_zero_treated_as_zero = True
            pdf_local_values = np.zeros_like(pdf_local_values, dtype=np.float64)
            pdf_value_range = (-0.5, 0.5)
            log_rank0(
                rank,
                f"  Normalization scale for {pdf_spec['pdf_name']} from {normalization_scale_field_name} is "
                f"{normalization_scale:.6g}, "
                "so the field is being treated as numerically zero for rank-stable PDF storage.",
            )

        pdf_result = compute_distributed_field_pdf(
            pdf_local_values,
            comm,
            bins=pdf_bins,
            value_range=pdf_value_range,
            normalization_scale=(1.0 if near_zero_treated_as_zero else normalization_scale),
            normalization_offset=(0.0 if near_zero_treated_as_zero else normalization_offset),
            pdf_name=pdf_spec["pdf_name"],
            source_field=source_field,
            normalization=normalization,
            plot_title=pdf_spec.get("plot_title", field_label),
            x_label=pdf_spec.get("x_label", latex_label),
            raw_x_label=pdf_spec.get("raw_x_label", field_label),
            y_label=pdf_spec.get("y_label", "PDF"),
        )
        if near_zero_treated_as_zero:
            pdf_result["measured_normalization_scale"] = float(normalization_scale)
            pdf_result["normalization_scale"] = 0.0
            pdf_result["normalization_offset"] = float(normalization_offset)
            pdf_result["near_zero_std_tolerance"] = float(NEAR_ZERO_PDF_STD_TOL)
            pdf_result["near_zero_field_treated_as_zero"] = True
            pdf_result["binning_warning"] = (
                f"{normalization_scale_field_name} normalization scale is <= {NEAR_ZERO_PDF_STD_TOL:.1e}, so the normalized PDF was stored "
                "as a zero-field delta approximation for rank stability."
            )
        pdf_result["source_field_mean"] = float(global_stats["global_mean"])
        pdf_result["source_field_std"] = float(global_stats["global_std"])
        pdf_result["source_field_rms"] = float(global_stats["global_rms"])
        pdf_result["source_field_min"] = float(global_stats["global_min"])
        pdf_result["source_field_max"] = float(global_stats["global_max"])
        if normalization_reference_field:
            pdf_result["normalization_reference_field"] = normalization_reference_field
        if normalization_reference_label:
            pdf_result["normalization_reference_label"] = normalization_reference_label
        if normalization_reference_std is not None:
            pdf_result["normalization_reference_std"] = float(normalization_reference_std)
        if normalization_reference_rms is not None:
            pdf_result["normalization_reference_rms"] = float(normalization_reference_rms)
        pdf_result["source_file"] = os.path.abspath(prepared_path)
        pdf_result["source_h5"] = os.path.abspath(prepared_path)
        pdf_result["step"] = str(meta["step"])
        pdf_result["time"] = float(meta["time"])
        pdf_result["grid_shape"] = np.asarray(meta["shape"], dtype=np.int64)

        if rank == 0:
            save_pdf_serial(slice_data_path, pdf_spec["pdf_name"], pdf_result)
            output_path = field_pdf_output_path(prepared_path, pdf_spec["pdf_name"], output_format=output_format)
            smooth_output_path = field_pdf_output_path(
                prepared_path,
                pdf_spec["pdf_name"],
                output_format=output_format,
                subdirectory="pdf_smooth",
            )
            plot_field_pdf(pdf_result, output_path, plot=plot, backend="yt")
            plot_smoothed_field_pdf(pdf_result, smooth_output_path, plot=plot, backend="yt")
            write_field_pdf_metadata(output_path, pdf_result)
            write_field_pdf_metadata(
                smooth_output_path,
                smooth_field_pdf_for_plot(pdf_result),
            )
            print_field_pdf_summary(pdf_result, output_path=output_path)
            log_rank0(rank, f"  Smoothed plot output: {smooth_output_path}")
            outputs.append(output_path)
            outputs.append(smooth_output_path)
        comm.Barrier()

    outputs = comm.bcast(outputs if rank == 0 else None, root=0)
    return outputs


def slice_metadata_log_path(output_path):
    """Return the append-only metadata log path for one slice_plots directory."""
    directory = os.path.dirname(os.path.abspath(output_path))
    return os.path.join(directory, "slice_render_metadata.txt")


def append_slice_metadata_log(
    metadata_path,
    source_file,
    field_label,
    axis,
    plane_index,
    plane_coord,
    step,
    time_value,
    plane_min,
    plane_max,
    global_min,
    global_max,
    colorbar_min,
    colorbar_max,
    global_rms,
    global_mean,
    value_normalization,
    output_path,
):
    """Append one rendered-slice metadata block to the shared slice-plots log."""
    lines = [
        "=" * 72,
        f"Source file: {os.path.abspath(source_file)}",
        f"Field: {field_label}",
        "",
        f"Axis: {axis}",
        f"Plane index: {int(plane_index)}",
        f"Plane coordinate: {float(plane_coord):.16e}",
        f"Step: {step}",
        f"Time: {float(time_value):.16e}",
        "",
        "Slice statistics:",
        f"Slice 2D min: {float(plane_min):.16e}",
        f"Slice 2D max: {float(plane_max):.16e}",
        "",
        "Global 3D field statistics:",
        f"Global 3D field min: {float(global_min):.16e}",
        f"Global 3D field max: {float(global_max):.16e}",
        f"Global 3D RMS normalization: {float(global_rms):.16e}",
        f"Global 3D average: {float(global_mean):.16e}",
        "",
        "Displayed colorbar limits:",
        f"Displayed colorbar min: {float(colorbar_min):.16e}",
        f"Displayed colorbar max: {float(colorbar_max):.16e}",
        "",
        f"Value normalization: {value_normalization}",
        f"Output: {os.path.abspath(output_path)}",
        "",
    ]
    with open(metadata_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def output_source_path(data_file, dataset_name, field_family):
    """Return the path whose stem should drive default output naming for one field."""
    if field_family not in {"scalar", "density_gradient"}:
        return data_file

    source_dataset_name = DENSITY_DATASET_NAME if field_family == "density_gradient" else dataset_name
    with h5py.File(data_file, "r") as hf:
        dataset = hf["fields"][source_dataset_name]
        source_path = dataset.attrs.get("source_path")
        source_h5 = dataset.attrs.get("source_h5")
        source_txt = dataset.attrs.get("source_txt")

    for candidate in (source_path, source_h5, source_txt):
        if candidate:
            return str(candidate)
    return data_file


def axis_bounds(coords):
    """Return lower edge, upper edge, and spacing for one axis coordinate array."""
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) == 1:
        delta = 1.0
    else:
        delta = float(coords[1] - coords[0])
    return float(coords[0]), float(coords[-1] + delta), delta


def build_yt_slice_dataset(plane, meta, axis, plane_index, field_label):
    """Construct a one-cell-thick yt dataset from one extracted 2D plane."""
    import yt

    plane = np.asarray(plane, dtype=np.float64)
    x0, x1, dx = axis_bounds(meta["x"])
    y0, y1, dy = axis_bounds(meta["y"])
    z0, z1, dz = axis_bounds(meta["z"])
    plane_coord = float(meta[axis][plane_index])

    if axis == "x":
        data = plane[np.newaxis, :, :]
        bbox = np.array([[plane_coord, plane_coord + dx], [y0, y1], [z0, z1]], dtype=np.float64)
    elif axis == "y":
        data = plane[:, np.newaxis, :]
        bbox = np.array([[x0, x1], [plane_coord, plane_coord + dy], [z0, z1]], dtype=np.float64)
    else:
        data = plane[:, :, np.newaxis]
        bbox = np.array([[x0, x1], [y0, y1], [plane_coord, plane_coord + dz]], dtype=np.float64)

    dataset = yt.load_uniform_grid(
        {field_label: data},
        data.shape,
        bbox=bbox,
        nprocs=1,
        periodicity=(False, False, False),
        unit_system="cgs",
    )
    return dataset, ("stream", field_label)


def render_plane_image(
    plane,
    meta,
    axis,
    plane_index,
    field_label,
    latex_label,
    cmap,
    width,
    output,
    plot,
    save_dpi,
    figure_size,
    colorbar_limits=None,
):
    """Render one plane to disk on rank 0 with yt SlicePlot."""
    import yt

    plane = np.asarray(plane, dtype=np.float64).copy()
    plane[np.abs(plane) < 1.0e-12] = 0.0
    plane = np.round(plane, decimals=10)
    plane_min = float(np.min(plane))
    plane_max = float(np.max(plane))
    # Slice images default to the normalized full 3D field limits so colorbars
    # stay comparable across different slice locations for the same variable.
    if colorbar_limits is None:
        zmin = plane_min
        zmax = plane_max
        print(f"  Using gathered 2D slice limits for colorbar: min={zmin:.6g}, max={zmax:.6g}", flush=True)
    else:
        zmin, zmax = (float(colorbar_limits[0]), float(colorbar_limits[1]))
        print(f"  Using stored global 3D limits for colorbar: min={zmin:.6g}, max={zmax:.6g}", flush=True)
        print(f"  Gathered 2D slice stats: min={plane_min:.6g}, max={plane_max:.6g}", flush=True)
    dataset, yt_field = build_yt_slice_dataset(plane, meta, axis, plane_index, field_label)
    slice_plot = yt.SlicePlot(dataset, axis, yt_field, center="c", origin="native")
    slice_plot.set_log(yt_field, False)
    slice_plot.set_cmap(yt_field, cmap)
    slice_plot.set_axes_unit("unitary")
    slice_plot.set_font(_yt_font_style())
    slice_plot.set_figure_size(float(figure_size))
    data_res = max(plane.shape)
    slice_plot.set_buff_size((data_res, data_res))
    if not np.isclose(zmin, zmax):
        slice_plot.set_zlim(yt_field, zmin=zmin, zmax=zmax)
    slice_plot.set_minorticks("all", False)
    slice_plot.set_colorbar_minorticks("all", False)
    if width is not None:
        slice_plot.set_width((float(width), "code_length"))

    slice_plot.render()
    window_plot = slice_plot.plots[yt_field]
    window_plot.cb.set_label(latex_label)
    image = window_plot.axes.images[0]
    image.set_interpolation("bicubic")
    if hasattr(image, "set_interpolation_stage"):
        image.set_interpolation_stage("rgba")
    saved_paths = [window_plot.save(output, mpl_kwargs={"dpi": save_dpi})]
    for saved_path in saved_paths:
        print(f"Saved: {saved_path}")
    if plot:
        slice_plot.show()
    return {
        "saved_paths": list(saved_paths),
        "plane_min": plane_min,
        "plane_max": plane_max,
        "colorbar_min": zmin,
        "colorbar_max": zmax,
    }


# Legacy matplotlib renderer reference kept for future use.
# import matplotlib.pyplot as plt
#
# def render_plane_image_matplotlib(
#     plane,
#     meta,
#     axis,
#     plane_index,
#     field_label,
#     latex_label,
#     cmap,
#     width,
#     output,
#     plot,
#     save_dpi,
#     figure_size,
# ):
#     plane = np.asarray(plane, dtype=np.float64).copy()
#     plane[np.abs(plane) < 1.0e-12] = 0.0
#     plane = np.round(plane, decimals=10)
#     info = plane_axes_and_extent(meta, axis)
#     with plt.rc_context(_mpl_plot_style()):
#         fig, ax = plt.subplots(figsize=(figure_size, figure_size))
#         image = ax.imshow(
#             plane,
#             origin="lower",
#             extent=info["extent"],
#             cmap=cmap,
#             interpolation="bicubic",
#             aspect="equal",
#         )
#         ax.set_box_aspect(1)
#         ax.set_xlabel(rf"${info['horizontal_name']}$", fontsize=24)
#         ax.set_ylabel(rf"${info['vertical_name']}$", fontsize=24)
#         apply_width(ax, meta, axis, width)
#         colorbar = fig.colorbar(image, ax=ax, label=latex_label)
#         fig.tight_layout()
#         fig.savefig(output, dpi=save_dpi)
#         if plot:
#             plt.show()
#         plt.close(fig)


def build_slice_requests(meta, slice_specs, default_axis):
    """Resolve one or more user slice specs into concrete requests."""
    requests = []
    if slice_specs:
        raw_specs = list(slice_specs)
        for spec in raw_specs:
            axis, selector = parse_slice_spec(spec)
            coords = meta[axis]
            plane_index, slice_tag = resolve_slice_index(coords, selector)
            requests.append((axis, plane_index, slice_tag))
        return requests

    default_requests = [
        ("z", ("center", None), "xy_center"),
        ("z", ("idx", 0), "xy_face"),
        ("x", ("idx", 0), "yz_face"),
        ("y", ("idx", 0), "zx_face"),
    ]
    seen = set()
    for axis, selector, default_tag in default_requests:
        coords = meta[axis]
        plane_index, _ = resolve_slice_index(coords, selector)
        key = (axis, plane_index)
        if key in seen:
            continue
        seen.add(key)
        requests.append((axis, plane_index, default_tag))
    return requests


def resolve_requested_fields(filepath, field_names):
    """Resolve requested field names against the available HDF5-backed field specs."""
    field_lookup = available_field_specs(filepath)
    requested_fields = finalize_requested_field_names(field_lookup, field_names)

    missing = [name for name in requested_fields if name not in field_lookup]
    if missing:
        available = ", ".join(sorted(field_lookup))
        raise ValueError(f"Unsupported field(s): {', '.join(missing)}. Available fields: {available}")

    return [field_lookup[name] for name in requested_fields]


def run_visualization(
    data_file,
    axis="z",
    field_names=None,
    cmap="RdBu_r",
    width=None,
    output=None,
    plot=False,
    comm=None,
    slice_specs=None,
    assume_structured_h5=False,
    backend_name="heffte_fftw",
    output_format="pdf",
    save_dpi=300,
    figure_size=8.0,
    save_slice_data=True,
    slice_data_output=None,
    value_normalization="none",
    thermo_gamma=1.4,
    pdf_only=False,
    pdf_bins=DEFAULT_FIELD_PDF_BINS,
    analysis_context=None,
):
    """Render one or more slice images from a structured HDF5 velocity file."""
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    prepared_path = data_file if assume_structured_h5 else ensure_structured_h5(data_file)
    requested_field_specs = resolve_requested_fields(prepared_path, field_names)
    all_available_field_specs = list(available_field_specs(prepared_path).values())
    pdf_specs = default_field_pdf_specs(all_available_field_specs, force_normalized_dilatation=pdf_only)
    field_specs = [] if pdf_only else requested_field_specs
    meta = read_grid_metadata(prepared_path)
    requests = [] if pdf_only else build_slice_requests(meta, slice_specs, axis)
    saved_slice_data_path = slice_data_output or default_slice_data_output(prepared_path)

    if pdf_only and output:
        raise ValueError("--output is not supported with --pdf-only because no slice image is rendered.")
    if output and (len(requests) != 1 or len(field_specs) != 1):
        raise ValueError("--output can only be used when rendering a single slice.")
    if pdf_only and not save_slice_data:
        raise ValueError("--pdf-only requires slice-data output because full-field PDFs are stored in the *_slices.h5 file.")

    needs_vorticity = any(spec[3] == "vorticity" for spec in field_specs)
    needs_divergence = any(spec[3] == "divergence" for spec in field_specs)
    needs_qcriterion = any(spec[3] == "qcriterion" for spec in field_specs)
    needs_rcriterion = any(spec[3] == "rcriterion" for spec in field_specs)
    needs_sound_speed = any(spec[0] == SOUND_SPEED_FIELD_NAME for spec in field_specs)
    needs_mach_number = any(spec[0] == MACH_NUMBER_FIELD_NAME for spec in field_specs)
    needs_density_gradient = any(spec[3] == "density_gradient" for spec in field_specs)
    needs_velocity_gradient_u12 = any(spec[3] == "velocity_gradient" for spec in field_specs)
    pdf_source_fields = {spec["source_field"] for spec in pdf_specs}
    needs_vorticity = needs_vorticity or ("vorticity_magnitude" in pdf_source_fields)
    needs_divergence = needs_divergence or ("div_u" in pdf_source_fields)
    needs_mach_number = needs_mach_number or (MACH_NUMBER_FIELD_NAME in pdf_source_fields)
    needs_velocity_gradient_u12 = needs_velocity_gradient_u12 or ("dux_dy" in pdf_source_fields)
    derived_cache = None
    stats_cache = {}
    if analysis_context is None and (
        needs_vorticity
        or needs_divergence
        or needs_qcriterion
        or needs_rcriterion
        or needs_sound_speed
        or needs_mach_number
        or needs_density_gradient
        or needs_velocity_gradient_u12
        or bool(pdf_specs)
    ):
        analysis_context = DistributedAnalysisContext.from_structured_h5(
            prepared_path,
            meta,
            comm,
            backend_name,
        )

    if rank == 0:
        print()
        print("-" * 60, flush=True)
        print("SLICE RENDERING", flush=True)
        print("-" * 60, flush=True)
        print(f"Loading {prepared_path} with {comm.Get_size()} MPI ranks...", flush=True)
        print(f"Structured domain dimensions: {meta['shape']}", flush=True)
        if pdf_only:
            print("PDF-only mode enabled: slice images will be skipped.", flush=True)
        else:
            print(f"Rendering {len(requests)} slice(s) for {len(field_specs)} field set(s)...", flush=True)
        if save_slice_data:
            print(f"Saving reusable slice data to: {saved_slice_data_path}", flush=True)
        print(flush=True)

    if save_slice_data:
        init_start = time.perf_counter()
        if rank == 0:
            initialize_slice_data_file(
                saved_slice_data_path,
                meta,
                field_specs,
                requests,
                data_file,
                prepared_path,
                backend_name,
                thermo_gamma=thermo_gamma,
            )
        comm.Barrier()
        log_rank0(rank, f"Initialized slice-data file in {time.perf_counter() - init_start:.2f}s")

    if (
        needs_vorticity
        or needs_divergence
        or needs_qcriterion
        or needs_rcriterion
        or needs_sound_speed
        or needs_mach_number
        or needs_density_gradient
        or needs_velocity_gradient_u12
    ):
        derived_start = time.perf_counter()
        derived_labels = _requested_derived_field_labels(
            needs_vorticity=needs_vorticity,
            needs_divergence=needs_divergence,
            needs_sound_speed=needs_sound_speed,
            needs_mach_number=needs_mach_number,
            needs_qcriterion=needs_qcriterion,
            needs_rcriterion=needs_rcriterion,
            needs_density_gradient=needs_density_gradient,
            needs_velocity_gradient_u12=needs_velocity_gradient_u12,
        )
        log_rank0(rank, "Derived-field stage:")
        log_rank0(rank, f"  Requested fields: {', '.join(derived_labels)}")
        if needs_sound_speed or needs_mach_number:
            log_rank0(rank, f"  Thermodynamic gamma: {thermo_gamma:.6g}")
        log_rank0(rank, "  Computing distributed derived fields...")
        derived_cache = compute_local_derived_fields(
            prepared_path,
            meta,
            comm,
            backend_name,
            include_vorticity=needs_vorticity,
            include_divergence=needs_divergence,
            include_qcriterion=needs_qcriterion,
            include_rcriterion=needs_rcriterion,
            include_sound_speed=needs_sound_speed,
            include_mach_number=needs_mach_number,
            include_density_gradient=needs_density_gradient,
            include_velocity_gradient_u12=needs_velocity_gradient_u12,
            density_dataset_name=(
                DENSITY_DATASET_NAME
                if (needs_density_gradient or needs_sound_speed or needs_mach_number)
                else None
            ),
            pressure_dataset_name=(
                PRESSURE_DATASET_NAME
                if (needs_sound_speed or needs_mach_number)
                else None
            ),
            thermo_gamma=thermo_gamma,
            analysis_context=analysis_context,
        )
        comm.Barrier()
        log_rank0(rank, f"  Completed in {time.perf_counter() - derived_start:.2f}s")
        log_rank0(rank, "")

    outputs = compute_and_store_full_field_pdfs(
        prepared_path,
        pdf_specs,
        stats_cache,
        derived_cache,
        meta,
        comm,
        save_slice_data=save_slice_data,
        slice_data_path=saved_slice_data_path,
        output_format=output_format,
        plot=plot,
        pdf_bins=pdf_bins,
        analysis_context=analysis_context,
    )
    if pdf_only:
        return outputs, (saved_slice_data_path if save_slice_data else None)

    # DANE can hang in the legacy MPI slice-data write path for x-normal slices,
    # where only one rank owns the requested plane. Keep the old branch below so
    # it can be restored easily if we ever want to revisit distributed slice writes.
    use_legacy_parallel_slice_data_write = False
    for dataset_name, field_label, latex_label, field_family in field_specs:
        global_stats = stats_cache.get(dataset_name)
        if global_stats is None:
            global_stats = compute_global_field_stats(
                prepared_path,
                dataset_name,
                field_family,
                meta,
                comm,
                derived_cache=derived_cache,
                analysis_context=analysis_context,
            )
            stats_cache[dataset_name] = global_stats
        if rank == 0 and field_family != "thermo":
            print()
            print(f"Field: {field_label}", flush=True)
            print_global_field_stats_block("  Global 3D statistics", global_stats)
        computed_rms = float(global_stats["global_rms"])
        field_value_normalization = str(value_normalization or "none").strip().lower()
        normalization_scale = 1.0
        if field_value_normalization == "global_rms":
            normalization_scale = computed_rms
            if normalization_scale <= 1.0e-30:
                field_value_normalization = "none"
                normalization_scale = 1.0
                log_rank0(rank, f"  RMS for {field_label} is ~0; leaving values unscaled.")
        elif field_value_normalization != "none":
            raise ValueError(
                f"Unsupported slice value normalization '{value_normalization}'. "
                "Use one of: ['none', 'global_rms']"
            )
        display_label = format_plot_label(latex_label, value_normalization=field_value_normalization)
        display_limits = (
            float(global_stats["global_min"]) / normalization_scale,
            float(global_stats["global_max"]) / normalization_scale,
        )
        for axis_name, plane_index, slice_tag in requests:
            slice_start = time.perf_counter()
            stored_slice_tag = storage_slice_tag(axis_name, slice_tag)
            if field_family in DERIVED_FIELD_FAMILIES:
                local_bounds, local_plane = extract_plane_from_boxes_local(
                    axis_name,
                    derived_cache["local_box"],
                    derived_cache[dataset_name],
                    plane_index,
                )
            else:
                local_bounds, local_plane = extract_plane_parallel_local(
                    prepared_path,
                    dataset_name,
                    axis_name,
                    plane_index,
                    meta,
                    comm,
                )

            gather_start = time.perf_counter()
            plane = gather_plane(axis_name, local_bounds, local_plane, meta["shape"], comm)
            local_plane_raw = None if local_plane is None else np.asarray(local_plane, dtype=np.float64)
            local_plane_display = None if local_plane_raw is None else local_plane_raw / normalization_scale
            if rank == 0:
                plane = np.asarray(plane, dtype=np.float64)
                plane_display = plane / normalization_scale
            else:
                plane_display = None
            log_rank0(
                rank,
                f"  Gather finished for {field_label}/{stored_slice_tag} in "
                f"{time.perf_counter() - gather_start:.2f}s",
            )
            if save_slice_data:
                write_start = time.perf_counter()
                if use_legacy_parallel_slice_data_write and h5py.get_config().mpi:
                    # Legacy MPI-HDF5 slice writes retained for reference. This path
                    # hung on DANE for x-normal slices when only one rank had data.
                    write_slice_plane_parallel(
                        saved_slice_data_path,
                        field_label,
                        stored_slice_tag,
                        axis_name,
                        local_bounds,
                        local_plane_raw,
                        comm,
                    )
                    comm.Barrier()
                    if rank == 0:
                        write_slice_stats_serial(
                            saved_slice_data_path,
                            field_label,
                            stored_slice_tag,
                            float(global_stats["global_min"]),
                            float(global_stats["global_max"]),
                            computed_rms,
                            float(global_stats["global_std"]),
                            float(global_stats["global_mean"]),
                            value_normalization="none",
                        )
                    comm.Barrier()
                else:
                    # Gathered planes are already needed for rank-0 rendering, so
                    # write the reusable slice-data file serially on rank 0.
                    if rank == 0:
                        write_slice_plane_serial(saved_slice_data_path, field_label, stored_slice_tag, plane)
                        write_slice_stats_serial(
                            saved_slice_data_path,
                            field_label,
                            stored_slice_tag,
                            float(global_stats["global_min"]),
                            float(global_stats["global_max"]),
                            computed_rms,
                            float(global_stats["global_std"]),
                            float(global_stats["global_mean"]),
                            value_normalization="none",
                        )
                    comm.Barrier()
                log_rank0(
                    rank,
                    f"  Slice-data write finished for {field_label}/{stored_slice_tag} in "
                    f"{time.perf_counter() - write_start:.2f}s",
                )

            rendered_paths = []
            if rank == 0:
                coord_value = meta[axis_name][plane_index]
                print()
                print(
                    f"  Slice normal={axis_name}, index={plane_index}, "
                    f"coord={coord_value:.6g}, step={meta['step']}, time={meta['time']:.6g}"
                ,
                    flush=True,
                )
                source_path = output_source_path(prepared_path, dataset_name, field_family)
                rendered = output or output_name(
                    source_path,
                    field_label,
                    axis_name,
                    slice_tag,
                    output_format,
                )
                render_start = time.perf_counter()
                rendered_info = render_plane_image(
                    plane_display,
                    meta,
                    axis_name,
                    plane_index,
                    field_label,
                    display_label,
                    cmap,
                    width,
                    rendered,
                    plot,
                    save_dpi,
                    figure_size,
                    colorbar_limits=display_limits,
                )
                rendered_paths = rendered_info["saved_paths"]
                for saved_path in rendered_paths:
                    append_slice_metadata_log(
                        slice_metadata_log_path(saved_path),
                        source_path,
                        field_label,
                        axis_name,
                        plane_index,
                        coord_value,
                        meta["step"],
                        meta["time"],
                        rendered_info["plane_min"],
                        rendered_info["plane_max"],
                        float(global_stats["global_min"]),
                        float(global_stats["global_max"]),
                        rendered_info["colorbar_min"],
                        rendered_info["colorbar_max"],
                        computed_rms,
                        float(global_stats["global_mean"]),
                        field_value_normalization,
                        saved_path,
                    )
                print(
                    f"  Render finished for {field_label}/{stored_slice_tag} in "
                    f"{time.perf_counter() - render_start:.2f}s",
                    flush=True,
                )
                outputs.extend(rendered_paths)
            rendered_paths = comm.bcast(rendered_paths, root=0)
            comm.Barrier()
            log_rank0(
                rank,
                f"  Completed slice {field_label}/{stored_slice_tag} in "
                f"{time.perf_counter() - slice_start:.2f}s",
            )
            if rank != 0:
                outputs.extend(rendered_paths)

    return outputs, (saved_slice_data_path if save_slice_data else None)


def main():
    parser = argparse.ArgumentParser(
        description="Render one or more velocity slices from TXT or structured HDF5",
        epilog=(
            "Examples:\n"
            "  python tools/visualize_velocity_yt.py data/SampledData0.h5 --field div_u --slice z:center\n"
            "  python tools/visualize_velocity_yt.py data/SampledData0.h5 --field div_u --norm global_rms\n"
            "  python tools/visualize_velocity_yt.py data/SampledData0.h5 --pdf-only --pdf-bins 128\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("data_file", help="Path to a SampledData .txt or structured .h5 file")
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        help="Slice spec axis:selector, e.g. z:center, y:idx=10, x:frac=0.25, z:coord=0.5",
    )
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"], help="Default center-slice axis if --slice is omitted")
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        help="Field to plot. Repeat to render multiple fields. Density-gradient magnitude is auto-appended when a density field exists.",
    )
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap")
    parser.add_argument("--width", type=float, default=None, help="Optional square plot width in domain units")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"], help="Output image format. Default is pdf.")
    parser.add_argument("--dpi", type=int, default=600, help="Raster save DPI. Default is 600.")
    parser.add_argument("--figsize", type=float, default=8.0, help="Square figure size in inches. Default is 8.0.")
    parser.add_argument("--output", "--out", dest="output", default=None, help="Optional output path for a single slice")
    parser.add_argument("--plot", action="store_true", help="Also display the plot on rank 0 after saving")
    parser.add_argument("--no-slice-data", action="store_true", help="Skip writing the combined *_slices.h5 file.")
    parser.add_argument("--slice-data-output", "--slice-data", dest="slice_data_output", default=None, help="Optional path for the combined slice-data HDF5 file.")
    parser.add_argument(
        "--value-normalization",
        "--norm",
        default="none",
        choices=["none", "global_rms"],
        help="Optional plot-time normalization for rendered slice values. Saved slice-data HDF5 values remain raw.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Skip slice-image rendering and only compute/store the configured full-field PDFs.",
    )
    parser.add_argument(
        "--pdf-bins",
        type=int,
        default=DEFAULT_FIELD_PDF_BINS,
        help=f"Number of bins for stored full-field PDFs. Default is {DEFAULT_FIELD_PDF_BINS}.",
    )
    parser.add_argument(
        "--backend",
        default="heffte_fftw",
        choices=["heffte_fftw", "heffte_stock"],
        help="HeFFTe backend used when deriving vorticity, Q/R, and density-gradient fields from FFTs.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.4,
        help="Ideal-gas ratio of specific heats used for derived sound-speed and Mach fields. Default is 1.4.",
    )
    args = parser.parse_args()
    run_visualization(
        args.data_file,
        axis=args.axis,
        field_names=args.field,
        cmap=args.cmap,
        width=args.width,
        output=args.output,
        plot=args.plot,
        slice_specs=args.slice,
        backend_name=args.backend,
        output_format=args.format,
        save_dpi=args.dpi,
        figure_size=args.figsize,
        save_slice_data=not args.no_slice_data,
        slice_data_output=args.slice_data_output,
        value_normalization=args.value_normalization,
        thermo_gamma=args.gamma,
        pdf_only=args.pdf_only,
        pdf_bins=args.pdf_bins,
    )
