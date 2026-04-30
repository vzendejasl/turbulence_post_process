"""Shared per-rank FFT analysis context with lazy cached derived fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import heffte
from .io import read_structured_local_dataset
from .io import read_structured_local_fields
from .layout import box_shape
from .layout import build_boxes
from .layout import choose_proc_grid
from .transform import backward_field
from .transform import forward_field
from .transform import get_backend
from .transform import local_wavenumber_mesh


@dataclass(frozen=True)
class AnalysisGrid:
    """Immutable grid metadata shared by one distributed analysis context."""

    shape: tuple[int, int, int]
    dx: float
    dy: float
    dz: float


class DistributedAnalysisContext:
    """Lazy per-rank cache for shared FFT intermediates and derived fields."""

    def __init__(
        self,
        *,
        filepath: str | None,
        grid: AnalysisGrid,
        comm,
        backend_name: str,
        local_box,
        local_vx,
        local_vy,
        local_vz,
    ) -> None:
        self.filepath = filepath
        self.grid = grid
        self.comm = comm
        self.backend_name = backend_name
        self.local_box = local_box
        self.local_shape = box_shape(local_box)
        self.global_points = int(np.prod(grid.shape))

        self.local_vx = np.asarray(local_vx, dtype=np.float64)
        self.local_vy = np.asarray(local_vy, dtype=np.float64)
        self.local_vz = np.asarray(local_vz, dtype=np.float64)

        self._backend = get_backend(backend_name)
        self._plan = heffte.fft3d(self._backend, local_box, local_box, comm)
        self._third_order_plan = None
        self._wavenumber_mesh = None
        self._k_squared = None
        self._nonzero_mask = None
        self._velocity_modes = None
        self._fluctuation_velocity_modes = None
        self._vorticity_components = None
        self._velocity_gradients = None
        self._dataset_cache: dict[str, np.ndarray] = {}
        self._derived_cache: dict[tuple[Any, ...], np.ndarray] = {}

    @classmethod
    def from_structured_h5(cls, filepath, meta, comm, backend_name):
        """Build one context by reading the local velocity slab from disk."""
        shape = tuple(int(value) for value in meta["shape"])
        proc_grid = choose_proc_grid(shape, comm.Get_size())
        boxes = build_boxes(shape, proc_grid)
        local_box = boxes[comm.Get_rank()]
        local_vx, local_vy, local_vz = read_structured_local_fields(filepath, local_box, comm)
        grid = AnalysisGrid(
            shape=shape,
            dx=float(meta["dx"]),
            dy=float(meta["dy"]),
            dz=float(meta["dz"]),
        )
        return cls(
            filepath=filepath,
            grid=grid,
            comm=comm,
            backend_name=backend_name,
            local_box=local_box,
            local_vx=local_vx,
            local_vy=local_vy,
            local_vz=local_vz,
        )

    @classmethod
    def from_local_velocity_fields(
        cls,
        *,
        filepath,
        shape,
        dx,
        dy,
        dz,
        comm,
        backend_name,
        local_box,
        local_vx,
        local_vy,
        local_vz,
    ):
        """Build one context from already-loaded local velocity fields."""
        return cls(
            filepath=filepath,
            grid=AnalysisGrid(
                shape=tuple(int(value) for value in shape),
                dx=float(dx),
                dy=float(dy),
                dz=float(dz),
            ),
            comm=comm,
            backend_name=backend_name,
            local_box=local_box,
            local_vx=local_vx,
            local_vy=local_vy,
            local_vz=local_vz,
        )

    @property
    def plan(self):
        return self._plan

    @property
    def third_order_plan(self):
        if self._third_order_plan is None:
            self._third_order_plan = heffte.fft3d(self._backend, self.local_box, self.local_box, self.comm)
        return self._third_order_plan

    @property
    def shape(self):
        return self.grid.shape

    @property
    def dx(self):
        return self.grid.dx

    @property
    def dy(self):
        return self.grid.dy

    @property
    def dz(self):
        return self.grid.dz

    def get_wavenumber_mesh(self):
        if self._wavenumber_mesh is None:
            self._wavenumber_mesh = local_wavenumber_mesh(
                self.shape,
                self.local_box,
                self.dx,
                self.dy,
                self.dz,
            )
        return self._wavenumber_mesh

    def get_k_squared(self):
        if self._k_squared is None:
            KX, KY, KZ = self.get_wavenumber_mesh()
            self._k_squared = KX**2 + KY**2 + KZ**2
        return self._k_squared

    def get_nonzero_mask(self):
        if self._nonzero_mask is None:
            self._nonzero_mask = self.get_k_squared() > 0.0
        return self._nonzero_mask

    def get_velocity_modes(self):
        if self._velocity_modes is None:
            self._velocity_modes = (
                forward_field(self.plan, self.local_vx).reshape(self.local_shape, order="C"),
                forward_field(self.plan, self.local_vy).reshape(self.local_shape, order="C"),
                forward_field(self.plan, self.local_vz).reshape(self.local_shape, order="C"),
            )
        return self._velocity_modes

    def get_fluctuation_velocity_modes(self):
        """Return cached velocity modes with the zero mode removed.

        Zeroing the k=0 mode is equivalent to subtracting the global mean in
        physical space, while avoiding an extra distributed real-space pass.
        """
        if self._fluctuation_velocity_modes is None:
            vx_k, vy_k, vz_k = self.get_velocity_modes()
            nonzero_mask = self.get_nonzero_mask()
            self._fluctuation_velocity_modes = (
                np.where(nonzero_mask, vx_k, 0.0 + 0.0j),
                np.where(nonzero_mask, vy_k, 0.0 + 0.0j),
                np.where(nonzero_mask, vz_k, 0.0 + 0.0j),
            )
        return self._fluctuation_velocity_modes

    def get_vorticity_components(self):
        if self._vorticity_components is None:
            vx_k, vy_k, vz_k = self.get_velocity_modes()
            KX, KY, KZ = self.get_wavenumber_mesh()
            omega_x_k = 1j * (KY * vz_k - KZ * vy_k)
            omega_y_k = 1j * (KZ * vx_k - KX * vz_k)
            omega_z_k = 1j * (KX * vy_k - KY * vx_k)
            self._vorticity_components = (
                backward_field(self.plan, omega_x_k, self.local_shape),
                backward_field(self.plan, omega_y_k, self.local_shape),
                backward_field(self.plan, omega_z_k, self.local_shape),
            )
        return self._vorticity_components

    def get_vorticity_magnitude(self):
        omega_x, omega_y, omega_z = self.get_vorticity_components()
        cache_key = ("vorticity_magnitude",)
        if cache_key not in self._derived_cache:
            self._derived_cache[cache_key] = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        return self._derived_cache[cache_key]

    def get_velocity_gradients(self):
        if self._velocity_gradients is None:
            vx_k, vy_k, vz_k = self.get_velocity_modes()
            KX, KY, KZ = self.get_wavenumber_mesh()
            self._velocity_gradients = {
                "dux_dx": backward_field(self.plan, 1j * KX * vx_k, self.local_shape),
                "dux_dy": backward_field(self.plan, 1j * KY * vx_k, self.local_shape),
                "dux_dz": backward_field(self.plan, 1j * KZ * vx_k, self.local_shape),
                "duy_dx": backward_field(self.plan, 1j * KX * vy_k, self.local_shape),
                "duy_dy": backward_field(self.plan, 1j * KY * vy_k, self.local_shape),
                "duy_dz": backward_field(self.plan, 1j * KZ * vy_k, self.local_shape),
                "duz_dx": backward_field(self.plan, 1j * KX * vz_k, self.local_shape),
                "duz_dy": backward_field(self.plan, 1j * KY * vz_k, self.local_shape),
                "duz_dz": backward_field(self.plan, 1j * KZ * vz_k, self.local_shape),
            }
        return self._velocity_gradients

    def get_divergence(self):
        gradients = self.get_velocity_gradients()
        cache_key = ("div_u",)
        if cache_key not in self._derived_cache:
            self._derived_cache[cache_key] = gradients["dux_dx"] + gradients["duy_dy"] + gradients["duz_dz"]
        return self._derived_cache[cache_key]

    def get_qcriterion(self):
        gradients = self.get_velocity_gradients()
        div_u = self.get_divergence()
        cache_key = ("q_criterion",)
        if cache_key not in self._derived_cache:
            trace_grad_u_sq = (
                gradients["dux_dx"] * gradients["dux_dx"]
                + gradients["dux_dy"] * gradients["duy_dx"]
                + gradients["dux_dz"] * gradients["duz_dx"]
                + gradients["duy_dx"] * gradients["dux_dy"]
                + gradients["duy_dy"] * gradients["duy_dy"]
                + gradients["duy_dz"] * gradients["duz_dy"]
                + gradients["duz_dx"] * gradients["dux_dz"]
                + gradients["duz_dy"] * gradients["duy_dz"]
                + gradients["duz_dz"] * gradients["duz_dz"]
            )
            self._derived_cache[cache_key] = 0.5 * (div_u**2 - trace_grad_u_sq)
        return self._derived_cache[cache_key]

    def get_rcriterion(self):
        gradients = self.get_velocity_gradients()
        cache_key = ("r_criterion",)
        if cache_key not in self._derived_cache:
            self._derived_cache[cache_key] = -(
                gradients["dux_dx"] * (gradients["duy_dy"] * gradients["duz_dz"] - gradients["duy_dz"] * gradients["duz_dy"])
                - gradients["dux_dy"] * (gradients["duy_dx"] * gradients["duz_dz"] - gradients["duy_dz"] * gradients["duz_dx"])
                + gradients["dux_dz"] * (gradients["duy_dx"] * gradients["duz_dy"] - gradients["duy_dy"] * gradients["duz_dx"])
            )
        return self._derived_cache[cache_key]

    def get_local_dataset(self, dataset_name):
        if dataset_name not in self._dataset_cache:
            if not self.filepath:
                raise ValueError(f"Dataset '{dataset_name}' is unavailable because no source filepath is attached.")
            self._dataset_cache[dataset_name] = read_structured_local_dataset(
                self.filepath,
                dataset_name,
                self.local_box,
                self.comm,
            )
        return self._dataset_cache[dataset_name]

    def get_density_gradient_magnitude(self, density_dataset_name):
        cache_key = ("density_gradient_magnitude", density_dataset_name)
        if cache_key not in self._derived_cache:
            local_density = self.get_local_dataset(density_dataset_name)
            density_k = forward_field(self.plan, local_density).reshape(self.local_shape, order="C")
            KX, KY, KZ = self.get_wavenumber_mesh()
            drho_dx = backward_field(self.plan, 1j * KX * density_k, self.local_shape)
            drho_dy = backward_field(self.plan, 1j * KY * density_k, self.local_shape)
            drho_dz = backward_field(self.plan, 1j * KZ * density_k, self.local_shape)
            self._derived_cache[cache_key] = np.sqrt(drho_dx**2 + drho_dy**2 + drho_dz**2)
        return self._derived_cache[cache_key]

    def get_sound_speed(self, density_dataset_name, pressure_dataset_name, thermo_gamma):
        cache_key = ("sound_speed", density_dataset_name, pressure_dataset_name, float(thermo_gamma))
        if cache_key not in self._derived_cache:
            local_density = self.get_local_dataset(density_dataset_name)
            local_pressure = self.get_local_dataset(pressure_dataset_name)
            invalid_density_count = self.comm.allreduce(int(np.count_nonzero(local_density <= 0.0)), op=self._mpi().SUM)
            if invalid_density_count:
                raise ValueError(
                    f"Cannot compute sound-speed and Mach fields: found {invalid_density_count} non-positive density value(s)."
                )
            local_sound_speed_sq = thermo_gamma * local_pressure / local_density
            invalid_sound_speed_sq_count = self.comm.allreduce(
                int(np.count_nonzero(local_sound_speed_sq < 0.0)),
                op=self._mpi().SUM,
            )
            if invalid_sound_speed_sq_count:
                raise ValueError(
                    f"Cannot compute sound-speed and Mach fields: found {invalid_sound_speed_sq_count} negative gamma*p/rho value(s)."
                )
            self._derived_cache[cache_key] = np.sqrt(np.maximum(local_sound_speed_sq, 0.0))
        return self._derived_cache[cache_key]

    def get_mach_number(self, density_dataset_name, pressure_dataset_name, thermo_gamma):
        cache_key = ("mach_number", density_dataset_name, pressure_dataset_name, float(thermo_gamma))
        if cache_key not in self._derived_cache:
            local_sound_speed = self.get_sound_speed(density_dataset_name, pressure_dataset_name, thermo_gamma)
            local_speed = np.sqrt(self.local_vx**2 + self.local_vy**2 + self.local_vz**2)
            sound_speed_floor = np.maximum(local_sound_speed, 1.0e-30)
            self._derived_cache[cache_key] = local_speed / sound_speed_floor
        return self._derived_cache[cache_key]

    @staticmethod
    def _mpi():
        from mpi4py import MPI

        return MPI
