"""Shared imports and small utilities for the FFT workflow."""

from __future__ import annotations

import math

import h5py
import numpy as np
from mpi4py import MPI

try:
    import heffte
except ImportError as exc:  # pragma: no cover - runtime environment specific
    raise SystemExit(
        "Unable to import heffte. Set PYTHONPATH to the HeFFTe Python wrapper "
        "install path before running this script."
    ) from exc


def global_mean_energy(vx, vy, vz, global_points, comm):
    local_sum = np.sum(vx**2 + vy**2 + vz**2, dtype=np.float64)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    return 0.5 * global_sum / float(global_points)


def global_range(values, comm):
    local_min = np.min(values)
    local_max = np.max(values)
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    return global_min, global_max


def global_mean(values, comm):
    local_values = np.asarray(values, dtype=np.float64)
    local_sum = float(np.sum(local_values, dtype=np.float64))
    local_count = int(local_values.size)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    if global_count <= 0:
        return 0.0
    return float(global_sum / float(global_count))


def global_rms(values, comm):
    local_values = np.asarray(values, dtype=np.float64)
    local_sum_sq = float(np.sum(local_values**2, dtype=np.float64))
    local_count = int(local_values.size)
    global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    if global_count <= 0:
        return 0.0
    return float(np.sqrt(global_sum_sq / float(global_count)))


def global_field_stats(values, comm):
    global_min, global_max = global_range(values, comm)
    return {
        "global_min": float(global_min),
        "global_max": float(global_max),
        "global_rms": global_rms(values, comm),
        "global_mean": global_mean(values, comm),
    }


def zero_near_zero(values, atol=1.0e-30):
    """Canonicalize roundoff-scale values so rank-count changes save identically."""
    arr = np.array(values, copy=True)
    arr[np.abs(arr) < atol] = 0.0
    return arr


def zero_near_zero_scalar(value, atol=1.0e-30):
    """Canonicalize scalar roundoff noise to an exact zero."""
    return 0.0 if abs(value) < atol else value
