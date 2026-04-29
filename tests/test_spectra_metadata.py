from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import types

import numpy as np


def _install_mpi4py_stub() -> None:
    """Provide a minimal mpi4py stub for metadata-writer unit tests."""
    try:
        __import__("mpi4py")
        return
    except ModuleNotFoundError:
        pass

    class _FakeComm:
        def Get_rank(self) -> int:
            return 0

        def Get_size(self) -> int:
            return 1

    class _FakeMPI:
        COMM_WORLD = _FakeComm()

        @staticmethod
        def Is_initialized() -> bool:
            return True

        @staticmethod
        def Init_thread() -> None:
            return None

    mpi4py_module = types.ModuleType("mpi4py")
    mpi4py_module.rc = types.SimpleNamespace(initialize=False, finalize=False)
    mpi4py_module.MPI = _FakeMPI
    sys.modules["mpi4py"] = mpi4py_module


_install_mpi4py_stub()

from postprocess_fft.io import save_spectra


class TestSpectraMetadata(unittest.TestCase):
    def test_thermodynamic_diagnostics_are_written_to_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spectra_metadata_") as tmpdir:
            input_path = Path(tmpdir) / "synthetic_case.h5"
            input_path.write_text("", encoding="utf-8")

            values = np.asarray([1.0, 2.0], dtype=np.float64)
            stats = {
                "global_min": 0.1,
                "global_max": 0.9,
                "global_rms": 0.5,
                "global_mean": 0.4,
            }
            turbulent_stats = {
                "global_min": 0.25,
                "global_max": 0.25,
                "global_rms": 0.25,
                "global_mean": 0.25,
            }
            turbulent_fluctuation_stats = {
                "global_min": 0.125,
                "global_max": 0.125,
                "global_rms": 0.125,
                "global_mean": 0.125,
            }
            mean_velocity_component_stats = {
                "mean_vx": 1.0e-6,
                "mean_vy": -2.0e-6,
                "mean_vz": 3.0e-6,
                "mean_speed_magnitude": float(np.sqrt(14.0e-12)),
            }

            save_spectra(
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                values,
                str(input_path),
                "42",
                1.25,
                4,
                4,
                4,
                3.5,
                1.0,
                2.5,
                0.75,
                thermo_gamma=1.4,
                sound_speed_stats=stats,
                mach_number_stats=stats,
                turbulent_mach_number_stats=turbulent_stats,
                turbulent_mach_fluctuation_stats=turbulent_fluctuation_stats,
                mean_velocity_component_stats=mean_velocity_component_stats,
            )

            metadata_path = input_path.with_name("synthetic_case_spectra_metadata.txt")
            metadata_text = metadata_path.read_text(encoding="utf-8")

            self.assertIn("# Thermodynamic gamma:", metadata_text)
            self.assertIn("# Sound speed stats:", metadata_text)
            self.assertIn("# Mach number stats:", metadata_text)
            self.assertIn("# Mean velocity components:", metadata_text)
            self.assertIn(
                "# Turbulent Mach number: Mt_raw = sqrt(2<KE>) / c_mean = 2.5000000000000000e-01",
                metadata_text,
            )
            self.assertIn(
                "# Turbulent Mach number (fluctuation-based): Mt_fluct = u'_rms / c_mean = 1.2500000000000000e-01",
                metadata_text,
            )


if __name__ == "__main__":
    unittest.main()
