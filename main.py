#!/usr/bin/env python3
"""Driver entrypoint for the mini turbulence post-processing workflow."""

from __future__ import annotations

import argparse
import os
import sys

from mpi4py import rc

rc.initialize = False
rc.finalize = False

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

from postprocess_fft.app import analyze_file_parallel
from postprocess_fft.io import plot_spectra
from postprocess_lib.auto_scalars import discover_auto_scalar_inputs
from postprocess_lib import converter
from postprocess_lib.prepare import ensure_all_structured_h5, resolve_existing_path
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
    parser.add_argument(
        "--qr-bins",
        type=int,
        default=256,
        help="Number of linear bins per axis for the Q-R joint PDF. Default is 256.",
    )
    parser.add_argument(
        "--structure-functions",
        action="store_true",
        help="Compute 2nd and 3rd order structure functions during the FFT step.",
    )
    parser.add_argument(
        "--structure-function-full-box",
        dest="structure_function_full_box",
        action="store_true",
        help="Sample axis-aligned structure functions over the full periodic box (r = 0..L). This is the default.",
    )
    parser.add_argument(
        "--structure-function-half-box",
        dest="structure_function_full_box",
        action="store_false",
        help="Sample axis-aligned structure functions over only the shortest periodic half-box (r = 0..L/2).",
    )
    parser.set_defaults(structure_function_full_box=True)
    parser.add_argument("--slice-axis", default="z", choices=["x", "y", "z"], help="Default slice normal axis")
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        help="Slice spec axis:selector, e.g. z:center, y:idx=10, x:frac=0.25, z:coord=0.5",
    )
    parser.add_argument(
        "--slice-field",
        action="append",
        default=[],
        help="Field for slice plots. Repeat to render multiple fields. Density-gradient magnitude is auto-appended when a density field exists.",
    )
    parser.add_argument(
        "--scalar-file",
        action="append",
        default=[],
        help="Optional extra sampled-data scalar TXT/HDF5 file to append. Density and pressure siblings are auto-discovered when present.",
    )
    parser.add_argument("--slice-cmap", default="RdBu_r", help="Colormap for slice plots")
    parser.add_argument("--slice-width", type=float, default=None, help="Optional square plot width in domain units")
    parser.add_argument("--slice-dpi", type=int, default=600, help="Slice image save DPI. Default is 600.")
    parser.add_argument(
        "--slice-value-normalization",
        default="none",
        choices=["none", "global_rms"],
        help="Optional plot-time normalization for rendered slice values. Saved slice-data HDF5 values remain raw. Default is none.",
    )
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
    parser.add_argument("--no-slice-data", action="store_true", help="Skip writing the combined *_slices.h5 file.")
    parser.add_argument(
        "--slice-data-output",
        default=None,
        help="Optional output path for the combined slice-data HDF5 file for a single input file.",
    )
    parser.add_argument("--slice-plot", action="store_true", help="Also display slice plots on rank 0")
    parser.add_argument(
        "--last-step",
        action="store_true",
        help="Only process the last write/snapshot from each Dedalus HDF5 file.",
    )
    # Experimental Dedalus x-block/VDS-source import is retained in
    # postprocess_lib.converter for future tuning, but the default runtime path
    # is back to the original whole-rank slab read.  Keep the CLI option hidden
    # for now so normal runs do not accidentally opt into the slower path.
    # parser.add_argument(
    #     "--dedalus-import-x-block-size",
    #     type=int,
    #     default=None,
    #     help=(
    #         "Number of x-planes per streaming read while importing Dedalus HDF5. "
    #         "Defaults to the TPP_DEDALUS_IMPORT_X_BLOCK_SIZE environment variable, "
    #         "or 1 when unset."
    #     ),
    # )
    parser.add_argument(
        "--keep-dedalus-import",
        action="store_true",
        help=(
            "Keep structured HDF5 files imported from Dedalus after this run. "
            "By default, imported Dedalus structured HDF5 files are deleted "
            "after FFT/slice processing finishes."
        ),
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    failures = 0
    prepared = []
    fft_results = []
    slice_outputs = []
    slice_data_outputs = []

    if rank == 0:
        print()
        print("=" * 72)
        print("MINI POST-PROCESSING PIPELINE")
        print("=" * 72)
        print(f"Processing {len(args.data_files)} file(s) with {size} MPI ranks...")

    if args.slice_output and len(args.data_files) != 1:
        raise SystemExit("--slice-output can only be used with a single input file.")
    if args.slice_data_output and len(args.data_files) != 1:
        raise SystemExit("--slice-data-output can only be used with a single input file.")
    if args.scalar_file and len(args.data_files) != 1:
        raise SystemExit("--scalar-file currently requires a single primary input file.")

    for idx, path in enumerate(args.data_files):
        try:
            resolved_input_path = resolve_existing_path(path)
            explicit_scalar_paths = [
                resolve_existing_path(scalar_path)
                for scalar_path in args.scalar_file
            ]
            auto_scalar_paths = discover_auto_scalar_inputs(
                resolved_input_path,
                explicit_scalar_paths=explicit_scalar_paths,
            )
            scalar_paths = []
            seen_scalar_paths = set()
            for scalar_path in explicit_scalar_paths + auto_scalar_paths:
                normalized_scalar_path = os.path.abspath(scalar_path)
                if normalized_scalar_path in seen_scalar_paths:
                    continue
                seen_scalar_paths.add(normalized_scalar_path)
                scalar_paths.append(scalar_path)

            if rank == 0:
                print()
                print("=" * 72)
                print(f"FILE {idx + 1}/{len(args.data_files)}")
                print("=" * 72)
                print(f"Input: {resolved_input_path}")
                if auto_scalar_paths:
                    print("Auto-discovered scalar inputs:")
                    for scalar_path in auto_scalar_paths:
                        print(f"  {scalar_path}")
            prepared_paths = ensure_all_structured_h5(
                resolved_input_path,
                last_only=args.last_step,
                include_origin=True,
            )

            for write_idx, prepared_item in enumerate(prepared_paths):
                prepared_path, imported_from_dedalus = prepared_item
                if rank == 0:
                    if len(prepared_paths) > 1:
                        print()
                        print(f"  --- Write {write_idx + 1}/{len(prepared_paths)} ---")
                    prepared.append(prepared_path)

                if scalar_paths:
                    added_scalar_fields = converter.append_scalar_fields_to_h5(
                        scalar_paths,
                        prepared_path,
                        create_scalar_h5=True,
                    )
                    if rank == 0 and added_scalar_fields:
                        print("Added scalar field datasets:")
                        for field_name in added_scalar_fields:
                            print(f"  {field_name}")

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
                        compute_structure_functions=args.structure_functions,
                        structure_function_full_domain=args.structure_function_full_box,
                        qr_joint_pdf_bins=args.qr_bins,
                    )
                    if rank == 0:
                        fft_results.append(fft_result)
                comm.Barrier()

                if not args.skip_slice:
                    if rank == 0:
                        print()
                    slice_output = args.slice_output if idx == 0 and write_idx == 0 else None
                    rendered, slice_data_path = run_visualization(
                        prepared_path,
                        axis=args.slice_axis,
                        field_names=args.slice_field,
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
                        save_slice_data=not args.no_slice_data,
                        value_normalization=args.slice_value_normalization,
                        slice_data_output=(
                            args.slice_data_output
                            if idx == 0 and write_idx == 0
                            else None
                        ),
                    )
                    if rank == 0:
                        slice_outputs.extend(rendered)
                        if slice_data_path is not None:
                            slice_data_outputs.append(slice_data_path)

                if imported_from_dedalus and not args.keep_dedalus_import:
                    comm.Barrier()
                    if rank == 0:
                        try:
                            os.remove(prepared_path)
                            print(f"Deleted imported Dedalus structured HDF5: {prepared_path}")
                        except FileNotFoundError:
                            print(f"Imported Dedalus structured HDF5 already absent: {prepared_path}")
                        except OSError as exc:
                            print(f"Warning: could not delete imported Dedalus structured HDF5 {prepared_path}: {exc}")
                    comm.Barrier()
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
            if slice_data_outputs:
                print("Slice data outputs:")
                for path in slice_data_outputs:
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
