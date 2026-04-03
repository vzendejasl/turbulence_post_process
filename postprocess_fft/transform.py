"""FFT plans, wavenumbers, and decomposition helpers."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from .common import heffte
from .common import global_range
from .layout import box_slices


def get_backend(backend_name):
    backend_name = backend_name.lower()
    backend_map = {
        "heffte_stock": heffte.backend.stock,
        "heffte_fftw": heffte.backend.fftw,
        "stock": heffte.backend.stock,
        "fftw": heffte.backend.fftw,
    }
    if backend_name not in backend_map:
        raise ValueError(
            f"Unsupported backend '{backend_name}'. "
            "Use one of: ['heffte_fftw', 'heffte_stock']"
        )

    if backend_name in ("fftw", "heffte_fftw") and not getattr(heffte.heffte_config, "enable_fftw", False):
        raise RuntimeError("HeFFTe was built without FFTW support.")
    return backend_map[backend_name]


def forward_field(plan, local_field):
    local_complex = np.empty(plan.size_outbox(), dtype=np.complex128)
    plan.forward(np.ascontiguousarray(local_field.ravel(order="C")), local_complex, heffte.scale.none)
    return local_complex


def backward_field(plan, local_field_k, local_shape):
    local_real = np.empty(plan.size_inbox(), dtype=np.float64)
    plan.backward(np.ascontiguousarray(local_field_k.ravel(order="C")), local_real, heffte.scale.full)
    return local_real.reshape(local_shape, order="C")


def local_wavenumber_mesh(shape, box, dx, dy, dz):
    nx, ny, nz = shape
    sx, sy, sz = box_slices(box)
    kx_phys = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)[sx]
    ky_phys = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)[sy]
    kz_phys = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)[sz]
    return np.meshgrid(kx_phys, ky_phys, kz_phys, indexing="ij")


def local_integer_wavenumber_mesh(shape, box):
    nx, ny, nz = shape
    sx, sy, sz = box_slices(box)
    kx_int = np.fft.fftfreq(nx, 1.0 / nx).astype(int)[sx]
    ky_int = np.fft.fftfreq(ny, 1.0 / ny).astype(int)[sy]
    kz_int = np.fft.fftfreq(nz, 1.0 / nz).astype(int)[sz]
    return np.meshgrid(kx_int, ky_int, kz_int, indexing="ij")


def print_component_ranges(name, vx, vy, vz, comm, root):
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    vx_rng = global_range(vx, comm)
    vy_rng = global_range(vy, comm)
    vz_rng = global_range(vz, comm)
    vm_rng = global_range(vmag, comm)
    if root:
        print(f"  {name}:")
        print(f"    vx: [{vx_rng[0]:.8f}, {vx_rng[1]:.8f}]")
        print(f"    vy: [{vy_rng[0]:.8f}, {vy_rng[1]:.8f}]")
        print(f"    vz: [{vz_rng[0]:.8f}, {vz_rng[1]:.8f}]")
        print(f"    |v|: [{vm_rng[0]:.8f}, {vm_rng[1]:.8f}]")


def verify_decomposition(plan, local_shape, KX, KY, KZ, vx_c_k, vy_c_k, vz_c_k, vx_r_k, vy_r_k, vz_r_k, comm, root):
    curl_c_x_k = 1j * (KY * vz_c_k - KZ * vy_c_k)
    curl_c_y_k = 1j * (KZ * vx_c_k - KX * vz_c_k)
    curl_c_z_k = 1j * (KX * vy_c_k - KY * vx_c_k)
    curl_c_x = backward_field(plan, curl_c_x_k, local_shape)
    curl_c_y = backward_field(plan, curl_c_y_k, local_shape)
    curl_c_z = backward_field(plan, curl_c_z_k, local_shape)
    curl_c_mag = np.sqrt(curl_c_x**2 + curl_c_y**2 + curl_c_z**2)

    div_r_k = 1j * (KX * vx_r_k + KY * vy_r_k + KZ * vz_r_k)
    div_r = backward_field(plan, div_r_k, local_shape)

    max_curl = comm.allreduce(np.max(np.abs(curl_c_mag)), op=MPI.MAX)
    max_div = comm.allreduce(np.max(np.abs(div_r)), op=MPI.MAX)

    if root:
        print("Verifying decomposition quality...")
        print(f"  Max |curl(v_compressive)|: {max_curl:.2e} (should be ~0)")
        print(f"  Max |div(v_rotational)|:  {max_div:.2e} (should be ~0)")
