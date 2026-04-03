#!/usr/bin/env python3
"""Driver entrypoint for the mini turbulence post-processing workflow."""

from __future__ import annotations

import argparse
import sys

from mpi4py import rc

rc.initialize = False
rc.finalize = False

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

from postprocess_fft.app import analyze_file_parallel
from postprocess_fft.io import plot_spectra
from postprocess_lib.prepare import ensure_structured_h5
from postprocess_vis.app import run_visualization


def main():
    parser = argparse.ArgumentParser(
        description="Mini turbulence post-processing driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mpirun -n 4 python main.py data/SampledData0.txt
  mpirun -n 4 python main.py data/SampledData0.h5
        """,
    )
    parser.add_argument("data_files", nargs="+", help="One or more .txt or structured .h5 inputs")
    parser.add_argument("--skip-fft", action="store_true", help="Skip the FFT/spectra step")
    parser.add_argument("--skip-slice", action="store_true", help="Skip the slice-plot step")
    parser.add_argument(
        "--backend",
        default="heffte_fftw",
        choices=["heffte_fftw", "heffte_stock"],
        help="HeFFTe backend to use internally. Default is HeFFTe with FFTW.",
    )
    parser.add_argument("--chunk-size", type=int, default=5_000_000, help="Chunk size for serial TXT fallback reads")
    parser.add_argument("--fft-plot", action="store_true", help="Plot FFT spectra on rank 0 after processing")
    parser.add_argument("--slice-axis", default="z", choices=["x", "y", "z"], help="Default slice normal axis")
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        help="Slice spec axis:selector, e.g. z:center, y:idx=10, x:frac=0.25, z:coord=0.5",
    )
    parser.add_argument(
        "--slice-field",
        default=None,
        choices=[
            "velocity_magnitude",
            "vx",
            "vy",
            "vz",
            "vorticity_magnitude",
            "wx",
            "wy",
            "wz",
        ],
        help="Field for slice plots. If omitted, render both velocity and vorticity magnitudes.",
    )
    parser.add_argument("--slice-cmap", default="RdBu_r", help="Colormap for slice plots")
    parser.add_argument("--slice-width", type=float, default=None, help="Optional square plot width in domain units")
    parser.add_argument("--slice-dpi", type=int, default=300, help="Slice image save DPI. Default is 300.")
    parser.add_argument(
        "--slice-figsize",
        type=float,
        default=8.0,
        help="Square slice figure size in inches. Default is 8.0.",
    )
    parser.add_argument(
        "--slice-format",
        default="pdf",
        choices=["pdf", "png"],
        help="Slice image format. Default is pdf.",
    )
    parser.add_argument(
        "--slice-output",
        default=None,
        help="Optional output path for a single slice from a single input file.",
    )
    parser.add_argument("--slice-plot", action="store_true", help="Also display slice plots on rank 0")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    failures = 0
    prepared = []
    fft_results = []
    slice_outputs = []

    if rank == 0:
        print()
        print("=" * 72)
        print("MINI POST-PROCESSING PIPELINE")
        print("=" * 72)
        print(f"Processing {len(args.data_files)} file(s) with {size} MPI ranks...")

    if args.slice_output and len(args.data_files) != 1:
        raise SystemExit("--slice-output can only be used with a single input file.")

    for idx, path in enumerate(args.data_files):
        try:
            if rank == 0:
                print()
                print("=" * 72)
                print(f"FILE {idx + 1}/{len(args.data_files)}")
                print("=" * 72)
                print(f"Input: {path}")
            prepared_path = ensure_structured_h5(path)
            if rank == 0:
                prepared.append(prepared_path)

            if not args.skip_fft:
                if rank == 0:
                    print()
                    print("-" * 60)
                    print("FFT / SPECTRA")
                    print("-" * 60)
                fft_result = analyze_file_parallel(
                    prepared_path,
                    comm,
                    header_lines=None,
                    chunk_size=args.chunk_size,
                    backend_name=args.backend,
                    visualize=False,
                )
                if rank == 0:
                    fft_results.append(fft_result)
            comm.Barrier()

            if not args.skip_slice:
                if rank == 0:
                    print()
                slice_output = args.slice_output if idx == 0 else None
                rendered = run_visualization(
                    prepared_path,
                    axis=args.slice_axis,
                    field_name=args.slice_field,
                    cmap=args.slice_cmap,
                    width=args.slice_width,
                    output=slice_output,
                    plot=args.slice_plot,
                    comm=comm,
                    slice_specs=args.slice,
                    assume_structured_h5=True,
                    backend_name=args.backend,
                    output_format=args.slice_format,
                    save_dpi=args.slice_dpi,
                    figure_size=args.slice_figsize,
                )
                if rank == 0:
                    slice_outputs.extend(rendered)
        except Exception as exc:
            if rank == 0:
                print(f"  CRITICAL ERROR processing {path}: {exc}")
            failures += 1
        comm.Barrier()

    if rank == 0:
        if fft_results and args.fft_plot:
            plot_spectra(fft_results)

        print("\n" + "=" * 40)
        if failures == 0:
            print(f"Pipeline completed successfully. {len(prepared)} file(s) processed.")
            print("Prepared HDF5 files:")
            for path in prepared:
                print(f"  {path}")
            if fft_results:
                print("FFT/spectra step completed.")
            if slice_outputs:
                print("Slice plot outputs:")
                for path in slice_outputs:
                    print(f"  {path}")
        else:
            print(f"Pipeline completed with ERRORS. {failures}/{len(args.data_files)} file(s) failed.")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 0
    finally:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()

    sys.exit(exit_code)
