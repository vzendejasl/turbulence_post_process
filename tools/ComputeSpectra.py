#!/usr/bin/env python3
"""Thin entrypoint for the distributed FFT post-processing application."""

from __future__ import annotations

import sys
from pathlib import Path
from mpi4py import rc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

rc.initialize = False
rc.finalize = False

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

from postprocess_fft.app import main


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 0
    finally:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()

    sys.exit(exit_code)
