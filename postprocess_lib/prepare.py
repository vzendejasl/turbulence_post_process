"""Input preparation helpers for the post-processing driver."""

from __future__ import annotations

import os

import h5py
from mpi4py import MPI

import convert_txt_to_hdf5 as converter


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def _broadcast_exists(path):
    exists = os.path.exists(path) if rank == 0 else False
    return comm.bcast(exists, root=0)


def validate_structured_h5(path):
    """Validate that an HDF5 file matches the FFT-ready structured schema."""
    with h5py.File(path, "r") as hf:
        return converter.is_structured_velocity_hdf5(hf)


def ensure_structured_h5(path):
    """
    Accept either TXT or HDF5 input and return a structured FFT-ready HDF5 path.

    TXT input uses the existing parallel converter workflow, preserving its
    print statements, verification, and automatic source-file deletion.
    """
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
        if not structured:
            raise ValueError(
                f"{path} is not a structured FFT-ready HDF5 file. "
                "Convert the TXT input first."
            )

        if rank == 0:
            print("  Input is already structured FFT-ready HDF5.")
            print(f"  Using existing HDF5: {path}")
        return path

    raise ValueError(f"Unsupported extension '{ext}'. Expected .txt or .h5.")

