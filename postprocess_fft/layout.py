"""Domain-decomposition helpers for distributed FFTs."""

from __future__ import annotations

import numpy as np

from .common import heffte


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


def choose_proc_grid(shape, nranks):
    """Choose a 3D processor grid that roughly minimizes halo surface area."""
    nx, ny, nz = shape
    best = None
    best_score = None
    for px in range(1, nranks + 1):
        if nranks % px != 0:
            continue
        rem = nranks // px
        for py in range(1, rem + 1):
            if rem % py != 0:
                continue
            pz = rem // py
            lx = nx / px
            ly = ny / py
            lz = nz / pz
            score = lx * ly + lx * lz + ly * lz
            if best_score is None or score < best_score:
                best = (px, py, pz)
                best_score = score
    return best


def build_boxes(shape, proc_grid):
    """Create rank-local boxes in the same rank order used by HeFFTe examples."""
    nx, ny, nz = shape
    px, py, pz = proc_grid
    row_major_order = np.array([2, 1, 0], dtype=np.int32)
    xs = split_axis(nx, px)
    ys = split_axis(ny, py)
    zs = split_axis(nz, pz)

    boxes = []
    for kz in range(pz):
        for ky in range(py):
            for kx in range(px):
                x0, x1 = xs[kx]
                y0, y1 = ys[ky]
                z0, z1 = zs[kz]
                boxes.append(
                    heffte.box3d(
                        [x0, y0, z0],
                        [x1 - 1, y1 - 1, z1 - 1],
                        row_major_order,
                    )
                )
    return boxes


def box_shape(box):
    return tuple(int(hi - lo + 1) for lo, hi in zip(box.low, box.high))


def box_slices(box):
    return tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(box.low, box.high))


def flatten_box(field, box):
    return np.ascontiguousarray(field[box_slices(box)].ravel(order="C"))


def scatter_field(field, boxes, comm):
    """Scatter one global field from rank 0 to local flattened arrays."""
    rank = comm.Get_rank()
    payload = None
    if rank == 0:
        payload = [flatten_box(field, box) for box in boxes]
    local = comm.scatter(payload, root=0)
    return np.ascontiguousarray(local, dtype=np.float64)

