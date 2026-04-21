"""Input preparation helpers for the post-processing driver."""

from __future__ import annotations

import os

import h5py
from mpi4py import MPI

from postprocess_lib import converter


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def _broadcast_exists(path):
    exists = os.path.exists(path) if rank == 0 else False
    return comm.bcast(exists, root=0)


def resolve_existing_path(path, preferred_extensions=(".h5", ".txt")):
    """
    Return an existing path, retrying with alternate extensions when needed.

    This is mainly a guard rail for users who pass `foo.txt` when `foo.h5`
    exists, or vice versa. If `path` has no extension, the preferred extension
    order is tried.
    """
    if _broadcast_exists(path):
        return path

    base, ext = os.path.splitext(path)
    ext = ext.lower()
    candidates = []

    if ext:
        for candidate_ext in preferred_extensions:
            if candidate_ext != ext:
                candidates.append(base + candidate_ext)
    else:
        for candidate_ext in preferred_extensions:
            candidates.append(path + candidate_ext)

    for candidate in candidates:
        if _broadcast_exists(candidate):
            if rank == 0:
                print(f"  Requested path not found: {path}")
                print(f"  Retrying with alternate extension: {candidate}")
            return candidate

    return path


def validate_structured_h5(path):
    """Validate that an HDF5 file matches the FFT-ready structured schema."""
    with h5py.File(path, "r") as hf:
        return converter.is_structured_velocity_hdf5(hf)


def validate_dedalus_field_h5(path):
    """Return True when an HDF5 file is a Dedalus field-output snapshot container."""
    with h5py.File(path, "r") as hf:
        return converter.is_dedalus_field_output_hdf5(hf)


def ensure_structured_h5(path, dedalus_import_x_block_size=None):
    """
    Accept either TXT or HDF5 input and return a structured FFT-ready HDF5 path.

    TXT input uses the existing parallel converter workflow, preserving its
    print statements, verification, and automatic source-file deletion.
    """
    path = resolve_existing_path(path, preferred_extensions=(".h5", ".txt"))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        success = converter.convert_file(path)
        if not success:
            raise RuntimeError(f"Conversion failed for {path}")
        return os.path.splitext(path)[0] + ".h5"

    if ext == ".h5":
        if rank == 0:
            print(f"\nProcessing: {path}")
        exists = _broadcast_exists(path)
        if not exists:
            raise FileNotFoundError(path)

        structured = validate_structured_h5(path) if rank == 0 else None
        structured = comm.bcast(structured, root=0)
        if structured:
            if rank == 0:
                print("  Input is already structured FFT-ready HDF5.")
                print(f"  Using existing HDF5: {path}")
            return path

        dedalus_field_output = validate_dedalus_field_h5(path) if rank == 0 else None
        dedalus_field_output = comm.bcast(dedalus_field_output, root=0)
        if dedalus_field_output:
            return converter.import_dedalus_snapshot_to_structured_h5(
                path,
                x_block_size=dedalus_import_x_block_size,
            )

        raise ValueError(
            f"{path} is not a structured FFT-ready HDF5 file or a supported Dedalus field-output HDF5 file."
        )

    raise ValueError(f"Unsupported extension '{ext}'. Expected .txt or .h5.")


def ensure_all_structured_h5(path, last_only=False, dedalus_import_x_block_size=None):
    """Like ensure_structured_h5 but imports all writes from multi-write Dedalus files.

    When *last_only* is True, only the last write from each Dedalus file is
    imported (the original single-snapshot behaviour).

    Returns a list of structured HDF5 paths.
    """
    path = resolve_existing_path(path, preferred_extensions=(".h5", ".txt"))
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        if last_only:
            raise SystemExit(
                f"--last-step is only supported for Dedalus HDF5 files, "
                f"but '{path}' is a TXT file."
            )
        success = converter.convert_file(path)
        if not success:
            raise RuntimeError(f"Conversion failed for {path}")
        return [os.path.splitext(path)[0] + ".h5"]

    if ext == ".h5":
        if rank == 0:
            print(f"\nProcessing: {path}")
        exists = _broadcast_exists(path)
        if not exists:
            raise FileNotFoundError(path)

        structured = validate_structured_h5(path) if rank == 0 else None
        structured = comm.bcast(structured, root=0)
        if structured:
            if last_only:
                raise SystemExit(
                    f"--last-step is only supported for Dedalus HDF5 files, "
                    f"but '{path}' is already a structured FFT-ready HDF5."
                )
            if rank == 0:
                print("  Input is already structured FFT-ready HDF5.")
                print(f"  Using existing HDF5: {path}")
            return [path]

        dedalus_field_output = validate_dedalus_field_h5(path) if rank == 0 else None
        dedalus_field_output = comm.bcast(dedalus_field_output, root=0)
        if dedalus_field_output:
            if last_only:
                return [
                    converter.import_dedalus_snapshot_to_structured_h5(
                        path,
                        x_block_size=dedalus_import_x_block_size,
                    )
                ]
            return converter.import_all_dedalus_snapshots_to_structured_h5(
                path,
                x_block_size=dedalus_import_x_block_size,
            )

        raise ValueError(
            f"{path} is not a structured FFT-ready HDF5 file "
            "or a supported Dedalus field-output HDF5 file."
        )

    raise ValueError(f"Unsupported extension '{ext}'. Expected .txt or .h5.")
