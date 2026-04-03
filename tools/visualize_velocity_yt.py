#!/usr/bin/env python3
"""Thin entrypoint for the scalable slice-visualization application."""

from __future__ import annotations

import sys

from mpi4py import rc

rc.initialize = False
rc.finalize = False

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

from postprocess_vis.app import main


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
