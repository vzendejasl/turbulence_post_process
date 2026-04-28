Python Scripts Setup and Run Guide
==================================

Scripts covered:
  main.py
  tools/ComputeSpectra.py
  tools/convert_txt_to_hdf5.py
  tools/plot_structure_function.py
  tools/visualize_velocity_yt.py
  tools/replot_slice_data.py

Repo layout:
  main.py
    Primary driver for the integrated pipeline.
  postprocess_lib/
    Shared input-preparation and TXT <-> HDF5 conversion library code.
  postprocess_fft/
    Shared HeFFTe/MPI spectra code used by the driver and the standalone FFT tool.
  postprocess_vis/
    Shared slice-visualization code used by the driver.
  tools/
    Standalone verification scripts that exercise pieces of the library independently.

Key workflow:
  1. Use main.py to accept TXT or HDF5 input.
  2. Convert MFEM TXT output to structured FFT-ready HDF5 when needed.
  3. Run the parallel spectra script on that HDF5 file.
  4. Optionally render one or more slices from the structured HDF5 file.
  5. Save the raw 2D slice data to one combined *_slices.h5 file for later replotting.
  6. Save a normalized Q-R joint PDF beside the spectra outputs.

Structured HDF5 schema written by tools/convert_txt_to_hdf5.py:
  /grid/x
  /grid/y
  /grid/z
  /fields/vx
  /fields/vy
  /fields/vz

This schema is what tools/ComputeSpectra.py and main.py read
directly in parallel.


================================================================================
LOCAL LINUX
================================================================================

This repo already contains a Linux-local setup based on the same requirements:

  environment.linux-mpi.yml
  setup_linux_heffte_env.sh

What it does:
  1. Creates a conda-forge environment that mirrors the working MPI side of the
     local dedalus_sim env:
       mpich 4.3.2
       fftw 3.3.10 mpi_mpich
       hdf5 1.14.6 mpi_mpich
       h5py 3.15.x mpi_mpich
       mpi4py 4.1.x
       yt 4.4.x
  2. Verifies h5py reports MPI support.
  3. Builds HeFFTe against the env's mpicc/mpicxx and FFTW install.
  4. Installs conda activation hooks so heffte stays on PYTHONPATH and
     LD_LIBRARY_PATH.

Create the env and build HeFFTe:

  bash setup_linux_heffte_env.sh

How yt gets installed:
  yt is listed directly in environment.linux-mpi.yml.
  setup_linux_heffte_env.sh uses that file with:
    conda env create -f environment.linux-mpi.yml
  or, if the env already exists:
    conda env update -n heffte-py-linux -f environment.linux-mpi.yml --prune

If the env already exists and you only want to add or refresh packages such as
yt, run:

  conda env update -n heffte-py-linux -f environment.linux-mpi.yml --prune

Activate:

  source /home/vzendejasl/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-linux

Sample data for a quick local smoke test lives at:

  data/SampledData0.txt

Verify:

  python -c "import h5py; print('h5py mpi =', h5py.get_config().mpi)"
  python -c "import heffte; print('HeFFTe =', heffte.__version__)"
  python -c "import heffte; print('FFTW =', heffte.heffte_config.enable_fftw)"
  python -c "import yt; print('yt =', yt.__version__)"

Run:

  mpirun -n 4 python main.py your_data.txt
  mpirun -n 4 python main.py your_data.h5
  mpirun -n 4 python main.py dedalus_fields_s2.h5
  mpirun -n 4 python tools/ComputeSpectra.py your_data.h5 --backend heffte_fftw --no-plot
  mpirun -n 4 python tools/convert_txt_to_hdf5.py your_data.txt
  mpirun -n 4 python tools/visualize_velocity_yt.py your_data.h5 --slice z:center --field vx

Tests:

  Compare spectra totals and full spectra between the sample TXT case and the
  reduced Dedalus HDF5 case:
    python -m tests.test_ke_spectra_compare

  Run the same comparison under MPI and print only one summary from rank 0:
    mpirun -n 4 python -m tests.test_ke_spectra_compare

  Check spectra rank consistency for both inputs across 1, 2, and 4 MPI ranks:
    python -m tests.test_rank_consistency

  The rank-consistency test is a serial launcher. It starts:
    mpirun -n 1 python -m tests.rank_consistency_worker ...
    mpirun -n 2 python -m tests.rank_consistency_worker ...
    mpirun -n 4 python -m tests.rank_consistency_worker ...

  Both tests compare spectra outputs only. They do not compare raw 3D fields
  directly, which avoids confusion from periodic duplicate endpoints in the
  sampled data.


--------------------------------------------------------------------------------
Testing
--------------------------------------------------------------------------------

Activate the environment first:

  source /home/vzendejasl/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-linux

Run the full test suite:

  python -m unittest discover -s tests -v

Run the TXT-vs-Dedalus spectra comparison test by itself:

  python -m tests.test_ke_spectra_compare

Optional MPI run for that same comparison test:

  mpirun -n 4 python -m tests.test_ke_spectra_compare

Run the rank-consistency test by itself:

  python -m tests.test_rank_consistency

What the tests do:

  tests.test_ke_spectra_compare
    - compares the sample-data case against the reduced Dedalus case
    - uses saved spectra outputs only
    - sums E_total to get total KE
    - sums Enstrophy to get total enstrophy
    - also reports full-spectrum relative L2 errors for E_total(k) and
      Enstrophy(k)
    - this is report-only; it does not enforce a pass/fail relative tolerance

  tests.test_rank_consistency
    - checks the same spectra workflow across 1, 2, and 4 MPI ranks
    - runs for both the sample fixture and the reduced Dedalus fixture
    - compares saved spectra outputs only
    - enforces:
        relative L2 tolerance = 1.0e-12
        relative sum tolerance = 1.0e-12

Important:
  - Run tests from the repo root.
  - Do not wrap `python -m unittest discover -s tests -v` in `mpirun`.
  - `tests.test_rank_consistency` launches its own 1/2/4-rank runs internally.
  - If `data/SampledData0.txt` is not present, the tests automatically fall back
    to `data/SampledData0.h5`.

Integrated pipeline:

  mpirun -n 4 python main.py your_data.txt

Main.py command patterns:

  Basic structured-HDF5 input:
    mpirun -n 4 python main.py your_velocity_data.h5

  Dedalus field-output HDF5 input:
    mpirun -n 4 python main.py your_dedalus_fields_s2.h5

  Import a large Dedalus field-output HDF5 file without running FFT or slices:
    mpirun -n 96 python main.py your_dedalus_fields_s9.h5 \
      --last-step \
      --skip-fft \
      --skip-slice

    The Dedalus import reads only /tasks/u into /fields/vx, /fields/vy, and
    /fields/vz.  The active import path uses the original whole-rank x-slab
    read, which is the faster path for the lower-rank Tuolumne runs.  The
    experimental x-block/direct-VDS importer is retained in
    postprocess_lib/converter.py as commented inactive code for future tuning,
    but it is not exposed as a command-line option.

    The temporary full-field structured HDF5 imported from Dedalus is deleted
    after FFT/slice work finishes by default.  To keep it for reuse or manual
    inspection, add:
      --keep-dedalus-import

    The default cleanup only deletes structured HDF5 files imported from
    Dedalus during the current run.  It does not delete TXT conversions,
    pre-existing structured HDF5 inputs, spectra outputs, Q-R PDF outputs,
    slice plots, or slice-data files.

  Basic TXT input:
    mpirun -n 4 python main.py your_velocity_data.txt

  FFT only:
    mpirun -n 4 python main.py your_velocity_data.h5 --skip-slice

  FFT with structure functions sampled over the default full periodic box:
    mpirun -n 4 python main.py your_velocity_data.h5 \
      --skip-slice \
      --structure-functions

  FFT with structure functions sampled over the half box only:
    mpirun -n 4 python main.py your_velocity_data.h5 \
      --skip-slice \
      --structure-functions \
      --structure-function-half-box

  Slice rendering only:
    mpirun -n 4 python main.py your_velocity_data.h5 --skip-fft

  Save PNG slices instead of PDF:
    mpirun -n 4 python main.py your_velocity_data.h5 --slice-format png

  Change slice DPI and figure size:
    mpirun -n 4 python main.py your_velocity_data.h5 --slice-dpi 600 --slice-figsize 10

  Render RMS-normalized slice plots while still saving raw slice arrays:
    mpirun -n 4 python main.py your_velocity_data.h5 --slice-value-normalization global_rms

  Render only selected fields:
    mpirun -n 4 python main.py your_velocity_data.h5 \
      --slice-field velocity_magnitude \
      --slice-field vorticity_magnitude

  Render Q-criterion and R-criterion slices explicitly:
    mpirun -n 4 python main.py your_velocity_data.h5 \
      --slice-field q_criterion

  Append scalar fields before slicing:
    mpirun -n 4 python main.py your_velocity_data.h5 \
      --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt \
      --scalar-file pressure_sampled_data_uniform_interpolated_cycle_0.txt

  Append three scalar fields in one run:
    mpirun -n 4 python main.py your_velocity_data.h5 \
      --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt \
      --scalar-file pressure_sampled_data_uniform_interpolated_cycle_0.txt \
      --scalar-file temperature_sampled_data_uniform_interpolated_cycle_0.txt

  Append scalars and render only selected outputs:
    mpirun -n 4 python main.py your_velocity_data.h5 \
      --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt \
      --slice-field velocity_magnitude \
      --slice-field density

What main.py now does:
  1. Accepts .txt or .h5 input.
  2. Converts TXT to structured FFT-ready HDF5 when needed.
  3. If the .h5 input is a Dedalus field-output file with /tasks/u, it imports the latest
     saved write into one structured FFT-ready snapshot named <input>_writeXXXXX.h5.
  4. Optionally appends one or more scalar sampled-data TXT files into /fields/<scalar_name>
     of the structured HDF5, and also writes one standalone scalar .h5 next to
     each scalar .txt input.
  5. Runs the FFT/spectra workflow on the resulting HDF5.
  6. Writes one or more slice PDFs after the FFT step.

Structure-function plotting:

  Compensated q2 on the default log-log axes:
    python tools/plot_structure_function.py -q2 your_spectra_structure_function.txt

  Compensated q3 on the default log-log axes:
    python tools/plot_structure_function.py -q3 your_spectra_structure_function.txt

  Compensated q2 on linear-linear axes:
    python tools/plot_structure_function.py -q2 your_spectra_structure_function.txt --plot-linear

  Compensated q3 on linear-linear axes:
    python tools/plot_structure_function.py -q3 your_spectra_structure_function.txt --plot-linear

  Uncompensated q2 on linear-linear axes:
    python tools/plot_structure_function.py -q2 your_spectra_structure_function.txt \
      --plot-linear \
      --uncompensated

  Uncompensated q3 on linear-linear axes:
    python tools/plot_structure_function.py -q3 your_spectra_structure_function.txt \
      --plot-linear \
      --uncompensated

  Hide the shell-averaged q3 curve:
    python tools/plot_structure_function.py -q3 your_spectra_structure_function.txt --no-shell

  Write to a specific PDF path:
    python tools/plot_structure_function.py -q2 your_spectra_structure_function.txt \
      --plot-linear \
      --output q2_structure_function_linear.pdf

Notes:
  - Full-box is now the default for newly computed axis-aligned structure functions.
  - If you want only the shortest periodic separations, add --structure-function-half-box during computation.
  - The plotter reads whatever r range is already stored in the file; it does not change half-box data into full-box data.
  - A full-box file can still be post-processed later to inspect only 0 <= r <= L/2 by keeping the first half of the saved rows.

Files written by the integrated pipeline:
  - the structured .h5 is written next to the original .txt input
  - Dedalus field-output input writes one structured snapshot named <input>_writeXXXXX.h5
    next to the original field file
  - the FFT spectra .txt and spectra metadata .txt are written next to that .h5 file
  - a Q-R joint PDF HDF5 file and PDF plot are written next to that .h5 file
  - the slice plots are written under slice_plots/ next to that .h5 file
  - the raw slice data are written under slice_data/ as one combined <base>_slices.h5 file

Dedalus notes:
  - The importer currently reads Dedalus field-handler HDF5 written with tasks/u.
  - It selects the latest saved write in the file by default.
  - Dedalus field files may use virtual datasets, so the importer uses independent HDF5
    reads on each MPI rank instead of collective MPI-IO for the source file.
  - The import is still parallel: each MPI rank reads only its own local x-slab from the
    Dedalus source file before the normal FFT and slice workflow runs.

Default slice outputs from main.py:
  - xy_center_velocity_magnitude.pdf
  - xy_face_velocity_magnitude.pdf
  - yz_face_velocity_magnitude.pdf
  - zx_face_velocity_magnitude.pdf
  - xy_center_vorticity_magnitude.pdf
  - xy_face_vorticity_magnitude.pdf
  - yz_face_vorticity_magnitude.pdf
  - zx_face_vorticity_magnitude.pdf
  - xy_center_q_criterion.pdf
  - xy_face_q_criterion.pdf
  - yz_face_q_criterion.pdf
  - zx_face_q_criterion.pdf
  - xy_center_r_criterion.pdf
  - xy_face_r_criterion.pdf
  - yz_face_r_criterion.pdf
  - zx_face_r_criterion.pdf
  - one combined slice_data/<base>_slices.h5 file containing all saved slice arrays

Default Q-R analysis outputs from the FFT step:
  - <base>_spectra.txt
  - <base>_spectra_metadata.txt
  - <base>_spectra_components.txt
  - <base>_spectra_qr_joint_pdf.h5

Spectra text output notes:
  - `k` remains the integer shell-center index used for the current shell binning
  - `k_phy` is the physical wavenumber associated with that shell center, computed as `(2*pi/L) * k`
  - `_comp` columns use integer-shell `k`
  - `_comp_phy` columns use `k_phy`
  - <base>_spectra_qr_joint_pdf.pdf

Slice output defaults:
  - format: pdf
  - save dpi: 600
  - square figure size: 8.0 inches

Canonical slice field names:
  - velocity_magnitude
  - vorticity_magnitude
  - vx, vy, vz
  - wx, wy, wz
  - q_criterion
  - r_criterion

The combined slice HDF5 stores:
  - all saved 2D slice arrays
  - horizontal and vertical coordinates for each slice
  - axis, plane index, plane coordinate, step, time, and source-file metadata
  - stored full-3D global min/max colorbar limits in the saved-value convention
  - stored full-3D RMS values so slices can be normalized later during replotting

This lets you replot slices later without recomputing the FFT/vorticity/Q-criterion/R-criterion workflow.

Q-R joint PDF normalization:
  - let A = grad(u) be the full velocity-gradient tensor
  - the code uses the full compressible invariants
    Q = 0.5 * ((tr(A))^2 - tr(A^2))
    R = -det(A)
  - the Frobenius norm used for nondimensionalization is
    |A|_F = sqrt(tr(A^T A))
  - Q is normalized locally as q_A = Q / |grad(u)|_F^2
  - R is normalized locally as r_A = R / |grad(u)|_F^3
  - the plotted axes are labeled with these nondimensional quantities:
    x-axis: r_A = R / |grad(u)|_F^3
    y-axis: q_A = Q / |grad(u)|_F^2
  - points are filtered before binning using
    |grad(u)|_F^2 / max(|grad(u)|_F^2) >= 1e-3
  - the saved HDF5 includes the bin edges, bin centers, raw counts, normalized joint PDF,
    total/retained sample counts, and the Frobenius-norm filter metadata
  - the default PDF plot uses bounds r_A in [-0.2, 0.2] and q_A in [-0.5, 0.5]
  - the default colorbar uses log-spaced PDF levels from 1e-3 to 1e2
  - the PDF plot overlays black enclosed-probability contours at 5%, 15%, 25%, and 50%
  - the PDF plot overlays a magenta contour enclosing the highest-density
    region that contains 90% of the total probability mass
  - by default the PDF figure is created with yt's PhasePlot/profile machinery
  - if yt is unavailable at runtime, the code falls back to the Matplotlib
    contour renderer automatically
  - set TURB_POSTPROCESS_QR_PLOT_BACKEND=matplotlib to force the fallback backend

Slice colorbar scaling:
  - velocity_magnitude, vorticity-based fields, q_criterion, r_criterion, and appended scalar fields
    use the full 3D global min/max by default
  - vx, vy, and vz continue to fall back to the gathered 2D slice limits
  - rendered slice values are unnormalized by default
  - pass --slice-value-normalization global_rms to normalize the rendered slice plots by the stored full-volume RMS
  - the saved slice-data HDF5 values remain raw even when the displayed plots use global_rms normalization

Optional scalar field inputs:
  - pass one or more sampled-data scalar files to main.py by repeating
    --scalar-file once per scalar input; each scalar path may be either .txt or .h5:
      --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt
      --scalar-file pressure_sampled_data_uniform_interpolated_cycle_0.txt
      --scalar-file temperature_sampled_data_uniform_interpolated_cycle_0.txt
  - the same alternate-extension fallback is used as for the primary input:
    if foo.txt is requested but only foo.h5 exists, it will use foo.h5, and vice versa
  - the scalar dataset names are parsed from the "Sampled Data, <name>" header line
  - each scalar is appended to /fields/<scalar_name> in the structured HDF5
  - if a scalar input is TXT, it is also converted to its own standalone HDF5
    file with the same basename, for example:
      density_sampled_data_uniform_interpolated_cycle_0.txt
      -> density_sampled_data_uniform_interpolated_cycle_0.h5
  - after appending, you can request those scalar names directly with --slice-field,
    for example:
      --slice-field density
      --slice-field pressure
      --slice-field temperature
  - the same default slices are then written for velocity, vorticity, Q, R, and all appended scalars
  - when both density and pressure are present, Mach number is also included in
    the default slice-field set
  - current restriction: when using --scalar-file, main.py expects one primary
    velocity input file per run

Multiple scalar files example:

  mpirun -n 4 python main.py velocity_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file pressure_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file temperature_sampled_data_uniform_interpolated_cycle_0.txt

This run will:
  - convert the primary velocity TXT to its structured HDF5 when needed
  - convert each scalar TXT to its own standalone scalar HDF5 when needed
  - append each scalar field into the primary structured HDF5 before slicing

Multiple scalar files with selected outputs only:

  mpirun -n 4 python main.py velocity_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file pressure_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file temperature_sampled_data_uniform_interpolated_cycle_0.txt \
    --slice-field velocity_magnitude \
    --slice-field density \
    --slice-field temperature

Important:
  tools/convert_txt_to_hdf5.py deletes the original .txt after a successful TXT -> HDF5
  conversion. Keep a copy if you want to preserve the ASCII file.


--------------------------------------------------------------------------------
Slice visualization
--------------------------------------------------------------------------------

yt is still included in the Linux conda environment, but the current scalable
slice workflow does not rebuild the full 3D volume in yt.  Instead it:

  1. Converts TXT to structured HDF5 once when needed
  2. Reads only the requested HDF5 plane(s) in parallel
  3. Gathers only the final 2D slice(s) to rank 0
  4. Writes the slice image files from rank 0

This is much more scalable for large runs and for requesting several slices.

The standalone visualization entrypoint accepts either:

  1. SampledData TXT input
  2. Structured HDF5 written by tools/convert_txt_to_hdf5.py

How it works:
  tools/visualize_velocity_yt.py detects the input type from the file extension.
  For .txt files it first runs the same TXT -> HDF5 conversion used by
  main.py, including the same verification prints and automatic deletion of
  the original TXT after a successful conversion.

  For .h5 files it reads the FFT-ready structured HDF5 schema directly in
  parallel and only loads the requested 2D plane(s).

Examples:

  python tools/visualize_velocity_yt.py data/SampledData0.h5
  mpirun -n 4 python tools/visualize_velocity_yt.py data/SampledData0.h5
  mpirun -n 4 python tools/visualize_velocity_yt.py data/SampledData0.txt

Slice-postprocessing command patterns:

  Default slice set from a structured HDF5 file:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.h5

  Default slice set from a TXT file:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.txt

  One field only:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.h5 --field vx

  One custom slice:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.h5 --slice z:center

  Several custom slices:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.h5 \
      --slice z:center \
      --slice x:center \
      --slice y:frac=0.25

  Save PNG output:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.h5 --format png

  Increase raster resolution:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.h5 --format png --dpi 600 --figsize 10

  Save only one selected slice to a specific path:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.h5 \
      --slice z:center \
      --field velocity_magnitude \
      --output my_slice.pdf

  Render Q-criterion explicitly:
    mpirun -n 4 python tools/visualize_velocity_yt.py your_velocity_data.h5 \
      --slice z:center \
      --field q_criterion

Multiple slices in one run:

  mpirun -n 4 python tools/visualize_velocity_yt.py data/SampledData0.h5 \
    --slice z:center \
    --slice x:center \
    --slice y:frac=0.25 \
    --field vorticity_magnitude

Supported slice selectors:
  center
  idx=<integer index>
  frac=<fraction from 0 to 1>
  coord=<physical coordinate value>

Optional interactive plotting:

  python tools/visualize_velocity_yt.py data/SampledData0.h5 --slice z:center --field vorticity_magnitude --plot
  python tools/visualize_velocity_yt.py data/SampledData0.h5 --format png --dpi 300 --figsize 8

Useful controls:
  - main.py:
      --slice-format {pdf,png}
      --slice-dpi 600
      --slice-figsize 10
      --scalar-file path/to/scalar_sampled_data.txt
      --slice-field velocity_magnitude
      --slice-field q_criterion
      --slice-field r_criterion
      --slice-field density
  - tools/visualize_velocity_yt.py:
      --format {pdf,png}
      --dpi 600
      --figsize 10

Output:

  data/slice_plots/SampledData0_xy_center_velocity_magnitude.pdf
  data/slice_plots/SampledData0_xy_center_vorticity_magnitude.pdf
  data/slice_plots/SampledData0_xy_center_q_criterion.pdf
  data/slice_plots/SampledData0_xy_center_r_criterion.pdf
  data/slice_data/SampledData0_slices.h5

Notes:
  - --plot only displays from rank 0.
  - On headless Linux sessions, --plot may not open a visible window even though
    the PNG is still written successfully.


--------------------------------------------------------------------------------
Replot saved slice data
--------------------------------------------------------------------------------

The slice workflow now also writes one combined HDF5 file:

  data/slice_data/SampledData0_slices.h5

That same file can now also store full-field PDFs under a top-level `pdfs/`
group.  The first built-in PDF is the normalized dilatation PDF:

  chi = div_u / rms(div_u)

The PDF currently uses a variable bin range taken from the global min/max of
the normalized field.  This is convenient for first-pass analysis, but it means
PDFs from different runs should be replotted onto a common range before making
direct comparisons.

Use the standalone reader/replot tool to inspect what is available:

  python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --list

Example replot:

  python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 \
    --field velocity_magnitude \
    --slice xy_center \
    --cmap viridis

Replot with RMS normalization applied from the saved metadata:

  python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 \
    --field velocity_magnitude \
    --slice xy_center \
    --value-normalization global_rms

Compute and store only the full-field PDFs during the slice stage:

  mpirun -n 4 python main.py your_velocity_data.h5 \
    --slice-pdf-only \
    --slice-pdf-bins 256

Inspect and replot stored full-field PDFs:

  python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 --list
  python tools/replot_field_pdf.py data/slice_data/SampledData0_slices.h5 \
    --pdf normalized_dilatation \
    --metadata

Example scalar workflow:

  mpirun -n 4 python main.py velocity_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file pressure_sampled_data_uniform_interpolated_cycle_0.txt

Useful controls:
  - --value-normalization {saved,none,global_rms}
  - --normalize
  - --vmin <value>
  - --vmax <value>
  - --width <domain width>
  - --output <path>
  - --format {png,pdf}

Optional yt adapter summary:

  python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 \
    --field velocity_magnitude \
    --slice xy_center \
    --yt-info


================================================================================
LOCAL MAC
================================================================================

Create env:

  /Users/victorzendejaslopez/anaconda3/bin/conda create -y -n heffte-py \
    -c conda-forge \
    python=3.11 cmake make cxx-compiler mpich mpi4py numpy scipy fftw git \
    pkg-config matplotlib pandas h5py

Activate:

  source /Users/victorzendejaslopez/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py

Build HeFFTe:

  git clone --depth 1 https://github.com/icl-utk-edu/heffte.git third_party/heffte

  cmake -S third_party/heffte -B third_party/heffte/build \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=/Users/victorzendejaslopez/Documents/MFEM/third_party/heffte/install \
    -D CMAKE_C_COMPILER=$CONDA_PREFIX/bin/mpicc \
    -D CMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/mpicxx \
    -D FFTW_ROOT=$CONDA_PREFIX \
    -D Heffte_ENABLE_FFTW=ON \
    -D Heffte_ENABLE_PYTHON=ON \
    -D Python_EXECUTABLE=$CONDA_PREFIX/bin/python

  cmake --build third_party/heffte/build -j4
  cmake --install third_party/heffte/build

Exports:

  export PYTHONPATH=/Users/victorzendejaslopez/Documents/MFEM/third_party/heffte/install/share/heffte/python:$PYTHONPATH

Verify:

  python -c "import heffte; print(heffte.__version__)"
  python -c "import h5py; print('h5py mpi =', h5py.get_config().mpi)"

Run:

  mpirun -n 4 python tools/ComputeSpectra.py \
    your_data.h5 --backend heffte_fftw --no-plot

  mpirun -n 4 python tools/convert_txt_to_hdf5.py your_data.txt


================================================================================
TUOLUMNE
================================================================================

This is the cleaned-up working path.

Important discovery:
  On Tuolumne, MPI-enabled h5py only worked correctly after mpi4py and h5py
  were both linked against the same GNU MPI stack.

Working linkage:
  mpi4py -> libmpi_gnu.so.12
  h5py   -> libmpi_gnu.so.12

Broken linkage we saw earlier:
  mpi4py -> libmpi_gnu.so.12
  h5py   -> libmpi_cray.so.12

That mismatch was the reason for:
  Attempting to use an MPI routine (internal_Comm_dup) before initializing or after finalizing MPICH


--------------------------------------------------------------------------------
1. Create the env
--------------------------------------------------------------------------------

  conda create -n heffte-py-tuo python=3.11 -y
  conda activate heffte-py-tuo
  conda install -c conda-forge numpy scipy matplotlib pandas cython cmake git -y


--------------------------------------------------------------------------------
2. Load modules
--------------------------------------------------------------------------------

Use the same stack for build and run:

  module --force purge
  module load StdEnv
  module load craype-x86-trento
  module load gcc/13.3.1
  module load cray-hdf5-parallel/1.14.3.7
  module load cray-fftw/3.3.10.11
  module load rocm/6.4.0
  module load cmake/3.29.2

Current Tuolumne note:
  The validated runtime stack on our recent Tuolumne tests used `StdEnv` plus
  `gcc/13.3.1` and did not require an explicit `cray-mpich` module load.
  `cray-mpich/9.1.0` is only available behind the `*-magic` compiler stack on
  some systems, so do not hard-code it unless your allocation actually supports
  that stack.


--------------------------------------------------------------------------------
3. Build mpi4py against GNU MPI
--------------------------------------------------------------------------------

  pip uninstall -y mpi4py
  MPICC=$(which mpicc) pip install --no-cache --force-reinstall \
    --no-build-isolation --no-binary=mpi4py mpi4py

Verify:

  ldd $(python -c "import mpi4py.MPI as m; print(m.__file__)") | egrep 'mpi|mpich'

Expected:
  libmpi_gnu.so.12


--------------------------------------------------------------------------------
4. Build MPI-enabled h5py against GNU parallel HDF5
--------------------------------------------------------------------------------

Use the GNU HDF5 variant under the Cray parallel HDF5 install:

  export HDF5_MPI=ON
  export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2
  export LD_LIBRARY_PATH=/opt/cray/pe/lib64/cce:$HDF5_DIR/lib:$LD_LIBRARY_PATH

  pip uninstall -y h5py
  env CC=$(which mpicc) \
      HDF5_MPI=ON \
      HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2 \
      pip install --no-cache --force-reinstall --no-build-isolation --no-binary=h5py h5py

Verify:

  python -c "import h5py; print(h5py.__version__)"
  python -c "import h5py; print(h5py.get_config().mpi)"
  ldd $(python -c "import h5py.h5 as h; print(h.__file__)") | egrep 'mpi|mpich|hdf5'

Expected:
  h5py.get_config().mpi -> True
  libmpi_gnu.so.12
  /opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2/lib/libhdf5.so.310

If h5py resolves to the conda env's own libhdf5*.so instead of the GNU Cray HDF5:
  move the conflicting conda HDF5 libraries aside and keep the GNU HDF5 path
  first on LD_LIBRARY_PATH.


--------------------------------------------------------------------------------
5. Build HeFFTe
--------------------------------------------------------------------------------

  export FFTW_PATH=/opt/cray/pe/fftw/3.3.10.11/x86_trento
  export MPICC=$(which mpicc)
  export MPICXX=$(which mpicxx)

  git clone --depth 1 https://github.com/icl-utk-edu/heffte.git third_party/heffte

  cmake -S third_party/heffte -B third_party/heffte/build \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=$PWD/third_party/heffte/install \
    -D CMAKE_C_COMPILER=${MPICC} \
    -D CMAKE_CXX_COMPILER=${MPICXX} \
    -D FFTW_ROOT=${FFTW_PATH} \
    -D Heffte_ENABLE_FFTW=ON \
    -D Heffte_ENABLE_PYTHON=ON \
    -D Python_EXECUTABLE=$(which python)

  cmake --build third_party/heffte/build -j8
  cmake --install third_party/heffte/build

If install fails because of missing CMakeRelink libheffte.so file, use:

  mkdir -p $PWD/third_party/heffte/install/lib64
  cp -P third_party/heffte/build/libheffte.so* $PWD/third_party/heffte/install/lib64/

Exports if HeFFTe is under the current working directory:

  export PYTHONPATH=$PWD/third_party/heffte/install/share/heffte/python:$PYTHONPATH
  export LD_LIBRARY_PATH=$PWD/third_party/heffte/install/lib64:$LD_LIBRARY_PATH

Exports for the installed path used on Tuolumne in our tests:

  export PYTHONPATH=/p/lustre5/zendejas/third_party/heffte/install/share/heffte/python:$PYTHONPATH
  export LD_LIBRARY_PATH=/p/lustre5/zendejas/third_party/heffte/install/lib64:$LD_LIBRARY_PATH

Verify:

  python -c "import heffte; print(heffte.__version__)"
  python -c "import heffte; print(heffte.heffte_config.libheffte_path)"


--------------------------------------------------------------------------------
6. Converter validation that worked
--------------------------------------------------------------------------------

After fixing mpi4py + h5py linkage, this worked on Tuolumne:

  srun -l -n 2 python ~/Documents/mfem_build/mfem/miniapps/fluids/navier/python_scripts/convert_txt_to_hdf5.py \
    blast_tgv3Dk2r3_SampledData/cycle_6358/velocity_sampled_data_uniform_interpolated_cycle_6358.txt

Observed result:
  - parallel TXT read worked
  - structured grid discovered correctly
  - parallel HDF5 write worked
  - parallel HDF5 verification read worked
  - TKE and row counts matched exactly


--------------------------------------------------------------------------------
7. Copy-paste batch script: converter
--------------------------------------------------------------------------------

  #!/bin/bash
  #SBATCH -N 1
  #SBATCH --ntasks-per-node 36
  #SBATCH -t 02:00:00

  module --force purge
  module load StdEnv
  module load craype-x86-trento
  module load gcc/13.3.1
  module load cray-hdf5-parallel/1.14.3.7
  module load cray-fftw/3.3.10.11
  module load rocm/6.4.0
  module load cmake/3.29.2

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-tuo

  export HDF5_MPI=ON
  export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2
  export PYTHONPATH=/p/lustre5/zendejas/third_party/heffte/install/share/heffte/python:$PYTHONPATH
  export LD_LIBRARY_PATH=/p/lustre5/zendejas/third_party/heffte/install/lib64:/opt/cray/pe/lib64/cce:$HDF5_DIR/lib:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1

  python -c "import h5py; print('h5py mpi =', h5py.get_config().mpi)"
  python -c "import mpi4py.MPI as m; print('mpi4py OK')"

  srun -l -n 36 python ~/Documents/mfem_build/mfem/miniapps/fluids/navier/python_scripts/convert_txt_to_hdf5.py \
    your_data.txt


--------------------------------------------------------------------------------
8. Copy-paste batch script: spectra
--------------------------------------------------------------------------------

  #!/bin/bash
  #SBATCH -N 2
  #SBATCH --ntasks-per-node 112
  #SBATCH -t 02:00:00

  module --force purge
  module load StdEnv
  module load craype-x86-trento
  module load gcc/13.3.1
  module load cray-hdf5-parallel/1.14.3.7
  module load cray-fftw/3.3.10.11
  module load rocm/6.4.0
  module load cmake/3.29.2

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-tuo

  export HDF5_MPI=ON
  export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2
  export PYTHONPATH=/p/lustre5/zendejas/third_party/heffte/install/share/heffte/python:$PYTHONPATH
  export LD_LIBRARY_PATH=/p/lustre5/zendejas/third_party/heffte/install/lib64:/opt/cray/pe/lib64/cce:$HDF5_DIR/lib:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1

  python -c "import h5py, heffte; print('h5py mpi =', h5py.get_config().mpi); print('HeFFTe =', heffte.__version__)"

  srun -l -n 224 python ~/Documents/mfem_build/mfem/miniapps/fluids/navier/python_scripts/tools/ComputeSpectra.py \
    your_data.h5 --backend heffte_fftw --no-plot


================================================================================
DANE
================================================================================

This is the validated working path on Dane for building the Python HeFFTe stack
in a personal conda environment and installing HeFFTe under:

  /p/lustre1/zendejas/third_party/heffte/install

Important discovery:
  On Dane, loading `fftw/3.3.10` switched the runtime from `openmpi/4.1.2` to
  `mvapich2/2.3.7`.  The working stack kept all build and runtime dependencies
  on the same MVAPICH2 toolchain:

  gcc/13.3.1
  mvapich2/2.3.7
  hdf5-parallel/1.14.0
  fftw/3.3.10
  cmake/3.30.5

Working linkage:
  mpi4py -> libmpi.so.12 from mvapich2-2.3.7-gcc-13.3.1
  h5py   -> libhdf5.so.310 from hdf5-1.14.0-mvapich2-2.3.7-gcc-13.3.1
  HeFFTe -> libheffte.so under /p/lustre1/zendejas/third_party/heffte/install/lib64


--------------------------------------------------------------------------------
1. Create the env
--------------------------------------------------------------------------------

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh

  conda create -y -n heffte-py-dane \
    python=3.11 \
    pip \
    numpy \
    scipy \
    matplotlib \
    pandas \
    yt \
    cython \
    cmake \
    ninja \
    pkg-config \
    git

  conda activate heffte-py-dane

Note:
  Do not install `mpi4py` or `h5py` from conda for this stack.  Build both with
  `pip` after the MPI/HDF5 modules are loaded so they link against Dane's
  MVAPICH2 and parallel HDF5 libraries.
  `yt` is required for slice plotting.  If you only want FFT/spectra, you can
  run `main.py ... --skip-slice` without `yt`.


--------------------------------------------------------------------------------
2. Load modules
--------------------------------------------------------------------------------

Use the same stack for build and run:

  module --force purge
  module load StdEnv
  module load gcc/13.3.1
  module load mvapich2/2.3.7
  module load hdf5-parallel/1.14.0
  module load fftw/3.3.10
  module load cmake/3.30.5


--------------------------------------------------------------------------------
3. Build mpi4py and MPI-enabled h5py
--------------------------------------------------------------------------------

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-dane

  export MPICC=$(which mpicc)
  export MPICXX=$(which mpicxx)
  export HDF5_DIR=$(dirname $(dirname $(which h5pcc)))

  python -m pip install --upgrade pip setuptools wheel
  python -m pip uninstall -y mpi4py h5py
  python -m pip install --no-cache-dir --force-reinstall \
    --no-build-isolation --no-binary=mpi4py mpi4py

  env CC=${MPICC} \
      HDF5_MPI=ON \
      HDF5_DIR=${HDF5_DIR} \
      python -m pip install --no-cache-dir --force-reinstall \
      --no-build-isolation --no-binary=h5py h5py

Verify:

  python -c "import mpi4py, mpi4py.MPI as MPI; print('mpi4py =', mpi4py.__version__); print(MPI.Get_library_version().splitlines()[0])"
  ldd $(python -c "import mpi4py.MPI as m; print(m.__file__)") | egrep 'mpi|mpich|mvapich'
  python -c "import h5py; print('h5py =', h5py.__version__); print('h5py mpi =', h5py.get_config().mpi)"
  ldd $(python -c "import h5py.h5 as h; print(h.__file__)") | egrep 'mpi|mpich|mvapich|hdf5'

Expected:
  - `mpi4py` reports MVAPICH2 2.3.7
  - `h5py.get_config().mpi` reports `True`
  - both `mpi4py` and `h5py` resolve to the MVAPICH2 and parallel HDF5 libraries
    under `/usr/tce/packages/...`


--------------------------------------------------------------------------------
4. Build HeFFTe
--------------------------------------------------------------------------------

Install HeFFTe in a writable path under `/p/lustre1/zendejas`:

  cd /p/lustre1/zendejas
  mkdir -p third_party

  export HEFFTE_ROOT=/p/lustre1/zendejas/third_party/heffte
  export HEFFTE_BUILD=/p/lustre1/zendejas/third_party/heffte/build
  export HEFFTE_INSTALL=/p/lustre1/zendejas/third_party/heffte/install

  export MPICC=$(which mpicc)
  export MPICXX=$(which mpicxx)
  export PYTHON_EXE=$(which python)
  export FFTW_ROOT=$(pkg-config --variable=prefix fftw3 2>/dev/null || dirname "$(dirname "$(which fftw-wisdom)")")

  git clone --depth 1 https://github.com/icl-utk-edu/heffte.git ${HEFFTE_ROOT}

  cmake -S ${HEFFTE_ROOT} -B ${HEFFTE_BUILD} \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=${HEFFTE_INSTALL} \
    -D CMAKE_C_COMPILER=${MPICC} \
    -D CMAKE_CXX_COMPILER=${MPICXX} \
    -D FFTW_ROOT=${FFTW_ROOT} \
    -D Heffte_ENABLE_FFTW=ON \
    -D Heffte_ENABLE_PYTHON=ON \
    -D Python_EXECUTABLE=${PYTHON_EXE}

  cmake --build ${HEFFTE_BUILD} -j8
  cmake --install ${HEFFTE_BUILD}

Exports:

  export PYTHONPATH=/p/lustre1/zendejas/third_party/heffte/install/share/heffte/python:${PYTHONPATH:-}
  export LD_LIBRARY_PATH=/p/lustre1/zendejas/third_party/heffte/install/lib64:/p/lustre1/zendejas/third_party/heffte/install/lib:/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-13.3.1/lib:${LD_LIBRARY_PATH:-}

Verify:

  python -c "import heffte; print('HeFFTe =', heffte.__version__)"
  python -c "import heffte; print('FFTW enabled =', heffte.heffte_config.enable_fftw)"
  python -c "import heffte; print('libheffte path =', heffte.heffte_config.libheffte_path)"
  python -c "import h5py; print('h5py mpi =', h5py.get_config().mpi)"
  python -c "import yt; print('yt =', yt.__version__)"


--------------------------------------------------------------------------------
5. One-time conda activation hook
--------------------------------------------------------------------------------

To avoid retyping the HeFFTe `PYTHONPATH` and `LD_LIBRARY_PATH` exports every
time, install this one-time activation hook while `heffte-py-dane` is active:

  conda activate heffte-py-dane
  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  cat > "$CONDA_PREFIX/etc/conda/activate.d/heffte.sh" <<'EOF'
  export PYTHONPATH=/p/lustre1/zendejas/third_party/heffte/install/share/heffte/python:${PYTHONPATH:-}
  export LD_LIBRARY_PATH=/p/lustre1/zendejas/third_party/heffte/install/lib64:/p/lustre1/zendejas/third_party/heffte/install/lib:/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-13.3.1/lib:${LD_LIBRARY_PATH:-}
  EOF

After that, each new shell only needs:

  module --force purge
  module load StdEnv
  module load gcc/13.3.1
  module load mvapich2/2.3.7
  module load hdf5-parallel/1.14.0
  module load fftw/3.3.10
  module load cmake/3.30.5
  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-dane


--------------------------------------------------------------------------------
6. One-time setup script
--------------------------------------------------------------------------------

This script is for the initial bootstrap only.  It creates the conda env,
rebuilds `mpi4py` and `h5py`, builds HeFFTe, and installs the one-time conda
activation hook.  You do not run this before every job.

  #!/bin/bash
  set -euo pipefail

  module --force purge
  module load StdEnv
  module load gcc/13.3.1
  module load mvapich2/2.3.7
  module load hdf5-parallel/1.14.0
  module load fftw/3.3.10
  module load cmake/3.30.5

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh

  conda create -y -n heffte-py-dane \
    python=3.11 \
    pip \
    numpy \
    scipy \
    matplotlib \
    pandas \
    yt \
    cython \
    cmake \
    ninja \
    pkg-config \
    git

  conda activate heffte-py-dane

  export MPICC=$(which mpicc)
  export MPICXX=$(which mpicxx)
  export HDF5_DIR=$(dirname $(dirname $(which h5pcc)))

  python -m pip install --upgrade pip setuptools wheel
  python -m pip uninstall -y mpi4py h5py
  python -m pip install --no-cache-dir --force-reinstall \
    --no-build-isolation --no-binary=mpi4py mpi4py

  env CC=${MPICC} \
      HDF5_MPI=ON \
      HDF5_DIR=${HDF5_DIR} \
      python -m pip install --no-cache-dir --force-reinstall \
      --no-build-isolation --no-binary=h5py h5py

  cd /p/lustre1/zendejas
  mkdir -p third_party

  export HEFFTE_ROOT=/p/lustre1/zendejas/third_party/heffte
  export HEFFTE_BUILD=/p/lustre1/zendejas/third_party/heffte/build
  export HEFFTE_INSTALL=/p/lustre1/zendejas/third_party/heffte/install
  export PYTHON_EXE=$(which python)
  export FFTW_ROOT=$(pkg-config --variable=prefix fftw3 2>/dev/null || dirname "$(dirname "$(which fftw-wisdom)"))

  if [ ! -d "${HEFFTE_ROOT}/.git" ]; then
    git clone --depth 1 https://github.com/icl-utk-edu/heffte.git "${HEFFTE_ROOT}"
  fi

  cmake -S "${HEFFTE_ROOT}" -B "${HEFFTE_BUILD}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX="${HEFFTE_INSTALL}" \
    -D CMAKE_C_COMPILER="${MPICC}" \
    -D CMAKE_CXX_COMPILER="${MPICXX}" \
    -D FFTW_ROOT="${FFTW_ROOT}" \
    -D Heffte_ENABLE_FFTW=ON \
    -D Heffte_ENABLE_PYTHON=ON \
    -D Python_EXECUTABLE="${PYTHON_EXE}"

  cmake --build "${HEFFTE_BUILD}" -j8
  cmake --install "${HEFFTE_BUILD}"

  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  cat > "$CONDA_PREFIX/etc/conda/activate.d/heffte.sh" <<'EOF'
  export PYTHONPATH=/p/lustre1/zendejas/third_party/heffte/install/share/heffte/python:${PYTHONPATH:-}
  export LD_LIBRARY_PATH=/p/lustre1/zendejas/third_party/heffte/install/lib64:/p/lustre1/zendejas/third_party/heffte/install/lib:/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-13.3.1/lib:${LD_LIBRARY_PATH:-}
  EOF

  export PYTHONPATH=/p/lustre1/zendejas/third_party/heffte/install/share/heffte/python:${PYTHONPATH:-}
  export LD_LIBRARY_PATH=/p/lustre1/zendejas/third_party/heffte/install/lib64:/p/lustre1/zendejas/third_party/heffte/install/lib:/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-13.3.1/lib:${LD_LIBRARY_PATH:-}

  python -c "import mpi4py, mpi4py.MPI as MPI; print('mpi4py =', mpi4py.__version__); print(MPI.Get_library_version().splitlines()[0])"
  python -c "import h5py; print('h5py =', h5py.__version__); print('h5py mpi =', h5py.get_config().mpi)"
  python -c "import yt; print('yt =', yt.__version__)"
  python -c "import heffte; print('HeFFTe =', heffte.__version__)"
  python -c "import heffte; print('FFTW enabled =', heffte.heffte_config.enable_fftw)"
  python -c "import heffte; print('libheffte path =', heffte.heffte_config.libheffte_path)"


--------------------------------------------------------------------------------
7. Interactive shell for normal use
--------------------------------------------------------------------------------

After the one-time setup is complete, each new shell only needs:

  module --force purge
  module load StdEnv
  module load gcc/13.3.1
  module load mvapich2/2.3.7
  module load hdf5-parallel/1.14.0
  module load fftw/3.3.10
  module load cmake/3.30.5
  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-dane

That is all.  Do not recreate the conda env for normal work or for jobs.
Do not run `set -euo pipefail` manually in an interactive shell.  That strict
mode is used in the batch scripts below, but it can make normal shell work and
tab-completion fragile.


--------------------------------------------------------------------------------
8. Copy-paste batch script: converter
--------------------------------------------------------------------------------

This is the script pattern to use for a real batch job after the env already
exists:

  #!/bin/bash
  #SBATCH -N 1
  #SBATCH --ntasks-per-node 36
  #SBATCH -t 02:00:00

  set -euo pipefail

  module --force purge
  module load StdEnv
  module load gcc/13.3.1
  module load mvapich2/2.3.7
  module load hdf5-parallel/1.14.0
  module load fftw/3.3.10
  module load cmake/3.30.5

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-dane

  export OMP_NUM_THREADS=1
  export POSTPROC_REPO=/g/g11/zendejas/turbulence_post_process

  python -c "import h5py; print('h5py mpi =', h5py.get_config().mpi)"
  python -c "import mpi4py.MPI as m; print('mpi4py OK')"
  python -c "import heffte; print('HeFFTe =', heffte.__version__)"
  python -c "import yt; print('yt =', yt.__version__)"

  srun -l -n 36 python "${POSTPROC_REPO}/tools/convert_txt_to_hdf5.py" your_data.txt


--------------------------------------------------------------------------------
9. Copy-paste batch script: spectra
--------------------------------------------------------------------------------

This is the script pattern to use for large FFT runs after the env already
exists:

  #!/bin/bash
  #SBATCH -N 2
  #SBATCH --ntasks-per-node 36
  #SBATCH -t 02:00:00

  set -euo pipefail

  module --force purge
  module load StdEnv
  module load gcc/13.3.1
  module load mvapich2/2.3.7
  module load hdf5-parallel/1.14.0
  module load fftw/3.3.10
  module load cmake/3.30.5

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-dane

  export OMP_NUM_THREADS=1
  export POSTPROC_REPO=/g/g11/zendejas/turbulence_post_process

  python -c "import h5py, heffte; print('h5py mpi =', h5py.get_config().mpi); print('HeFFTe =', heffte.__version__)"
  python -c "import yt; print('yt =', yt.__version__)"

  srun -l -n 72 python "${POSTPROC_REPO}/tools/ComputeSpectra.py" your_data.h5 --backend heffte_fftw --no-plot


================================================================================
NOTES
================================================================================

tools/convert_txt_to_hdf5.py
  - Reads the TXT file in parallel after rank 0 builds the chunk index.
  - Writes structured FFT-ready HDF5.
  - Verifies TKE and row counts after writing.
  - H5 -> TXT conversion remains serial on rank 0.

tools/ComputeSpectra.py
  - For the structured HDF5 schema above, each rank reads its local HDF5 slab directly.
  - FFTs and spectral decomposition are distributed through HeFFTe/MPI.
  - Legacy flat HDF5/TXT still uses rank 0 reconstruction plus scatter.

================================================================================
RZADAMS
================================================================================


## Final outcome

| Component | Status |
|---|---|
| Conda env in workspace | built |
| `mpi4py` | built against Cray MPICH |
| `h5py` | built with MPI enabled |
| HeFFTe | built and importable |
| FFTW backend in HeFFTe | enabled |

---

# 1. Create and activate the conda env in workspace

```bash
conda create -y -p /usr/workspace/zendejas/turbulence_post_process/.conda/envs/heffte-py-rzadams python=3.11
conda activate /usr/workspace/zendejas/turbulence_post_process/.conda/envs/heffte-py-rzadams
```

Install base packages:

```bash
conda install -c conda-forge numpy scipy matplotlib pandas cython cmake git pip -y
```

Optional, if you want plotting support:

```bash
conda install -c conda-forge yt -y
```

---

# 2. Load the working module stack

This was the working build stack:

```bash
module --force purge
module load PrgEnv-gnu/8.7.0
module load gcc/12.2.0
module load cmake/3.29.2
module load craype-x86-trento
module load cray-fftw/3.3.10.11
```

Notes:

| Module | Why |
|---|---|
| `PrgEnv-gnu/8.7.0` | required by Cray compiler wrappers |
| `gcc/12.2.0` | matched the GNU stack we used |
| `cmake/3.29.2` | for HeFFTe build |
| `craype-x86-trento` | sets target for this machine family |
| `cray-fftw/3.3.10.11` | FFTW for HeFFTe |

Also loaded implicitly through `PrgEnv-gnu`:

| Implicit module | Purpose |
|---|---|
| `cray-mpich/8.1.25` | MPI |
| `cray-libsci/21.08.1.2` | math libs |

---

# 3. Export the critical library paths

These exports were needed for successful builds:

```bash
export HDF5_MPI=ON
export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/cray/20.0
export PMI_LIBDIR=/opt/cray/pe/lib64
export CCE_LIBDIR=/opt/cray/pe/lib64/cce
export LIBRARY_PATH=${CCE_LIBDIR}:${PMI_LIBDIR}:${HDF5_DIR}/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${CCE_LIBDIR}:${PMI_LIBDIR}:${HDF5_DIR}/lib:${LD_LIBRARY_PATH:-}
export LDFLAGS="-L${CCE_LIBDIR} -L${PMI_LIBDIR} -L${HDF5_DIR}/lib ${LDFLAGS:-}"
```

These fixed:

| Problem | Fix |
|---|---|
| missing `libpmi.so` / `libpmi2.so` | `/opt/cray/pe/lib64` |
| missing `libmodules.so.1` | `/opt/cray/pe/lib64/cce` |
| HDF5 runtime loading | `HDF5_DIR` and HDF5 lib path |

---

# 4. Build `mpi4py`

We used the Cray compiler wrapper from the active module stack:

```bash
python -m pip uninstall -y mpi4py
MPICC=$(which cc) \
python -m pip install --no-cache-dir --force-reinstall \
  --no-build-isolation --no-binary=mpi4py mpi4py
```

Verification:

```bash
python -c "import mpi4py, mpi4py.MPI as MPI; print('mpi4py =', mpi4py.__version__); print(MPI.Get_library_version().splitlines()[0])"
ldd $(python -c "import mpi4py.MPI as m; print(m.__file__)") | egrep 'mpi|mpich|pmi|cray'
```

Expected result you got:

| Check | Result |
|---|---|
| `mpi4py` version | `4.1.1` |
| MPI runtime | `CRAY MPICH version 8.1.12.35` |

---

# 5. Build MPI-enabled `h5py`

We built `h5py` from source against Cray parallel HDF5, and used `--no-deps` so pip would not replace the already-working `mpi4py`.

```bash
python -m pip uninstall -y h5py
env CC=$(which cc) \
    HDF5_MPI=ON \
    HDF5_DIR=${HDF5_DIR} \
    python -m pip install --no-cache-dir --force-reinstall \
    --no-build-isolation --no-binary=h5py --no-deps h5py
```

Verification:

```bash
python -c "import h5py; print('h5py =', h5py.__version__); print('h5py mpi =', h5py.get_config().mpi)"
ldd $(python -c "import h5py.h5 as h; print(h.__file__)") | egrep 'mpi|mpich|pmi|hdf5|cray|modules'
```

Expected result you got:

| Check | Result |
|---|---|
| `h5py` version | `3.16.0` |
| `h5py.get_config().mpi` | `True` |

---

# 6. Build HeFFTe

We used `x86_milan` as the FFTW root because `craype-x86-trento` was loaded, but no `x86_trento` FFTW directory existed under `/opt/cray/pe/fftw/3.3.10.11`.

Set build variables:

```bash
cd /usr/workspace/zendejas/turbulence_post_process

export MPICC=$(which cc)
export MPICXX=$(which CC)
export PYTHON_EXE=$(which python)
export FFTW_ROOT=/opt/cray/pe/fftw/3.3.10.11/x86_milan
```

Clone HeFFTe if needed:

```bash
git clone --depth 1 https://github.com/icl-utk-edu/heffte.git third_party/heffte
```

Configure:

```bash
cmake -S third_party/heffte -B third_party/heffte/build \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_SHARED_LIBS=ON \
  -D CMAKE_INSTALL_PREFIX=$PWD/third_party/heffte/install \
  -D CMAKE_C_COMPILER=${MPICC} \
  -D CMAKE_CXX_COMPILER=${MPICXX} \
  -D FFTW_ROOT=${FFTW_ROOT} \
  -D Heffte_ENABLE_FFTW=ON \
  -D Heffte_ENABLE_PYTHON=ON \
  -D Python_EXECUTABLE=${PYTHON_EXE}
```

Build and install:

```bash
cmake --build third_party/heffte/build -j8
cmake --install third_party/heffte/build
```

If needed, the fallback install copy command was:

```bash
mkdir -p $PWD/third_party/heffte/install/lib64
cp -P third_party/heffte/build/libheffte.so* $PWD/third_party/heffte/install/lib64/
```

---

# 7. Export HeFFTe runtime paths

```bash
export PYTHONPATH=$PWD/third_party/heffte/install/share/heffte/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$PWD/third_party/heffte/install/lib64:$PWD/third_party/heffte/install/lib:${CCE_LIBDIR}:${PMI_LIBDIR}:${HDF5_DIR}/lib:${LD_LIBRARY_PATH:-}
```

Verification:

```bash
python -c "import heffte; print('HeFFTe =', heffte.__version__)"
python -c "import heffte; print('FFTW enabled =', heffte.heffte_config.enable_fftw)"
python -c "import heffte; print('libheffte path =', heffte.heffte_config.libheffte_path)"
```

Expected result you got:

| Check | Result |
|---|---|
| HeFFTe version | `2.4.1` |
| FFTW enabled | `True` |
| lib path | `/usr/workspace/zendejas/turbulence_post_process/third_party/heffte/install/lib64/libheffte.so` |

---

# 8. Recommended reusable setup script

Save this as `env_rzadams.sh` in your repo:

```bash
#!/bin/bash

module --force purge
module load PrgEnv-gnu/8.7.0
module load gcc/12.2.0
module load cmake/3.29.2
module load craype-x86-trento
module load cray-fftw/3.3.10.11

source /collab/usr/gapps/python/toss_4_x86_64_ib/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate /usr/workspace/zendejas/turbulence_post_process/.conda/envs/heffte-py-rzadams

export HDF5_MPI=ON
export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/cray/20.0
export PMI_LIBDIR=/opt/cray/pe/lib64
export CCE_LIBDIR=/opt/cray/pe/lib64/cce

export PYTHONPATH=/usr/workspace/zendejas/turbulence_post_process/third_party/heffte/install/share/heffte/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/workspace/zendejas/turbulence_post_process/third_party/heffte/install/lib64:/usr/workspace/zendejas/turbulence_post_process/third_party/heffte/install/lib:${CCE_LIBDIR}:${PMI_LIBDIR}:${HDF5_DIR}/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${CCE_LIBDIR}:${PMI_LIBDIR}:${HDF5_DIR}/lib:${LIBRARY_PATH:-}
export LDFLAGS="-L${CCE_LIBDIR} -L${PMI_LIBDIR} -L${HDF5_DIR}/lib ${LDFLAGS:-}"
export OMP_NUM_THREADS=1
```

Use it with:

```bash
source env_rzadams.sh
```

---

# 9. Runtime launcher note

From your latest shell, `srun` disappeared after module purge, but `flux` works.

So for runtime testing on rzadams, use **Flux** unless your site tells you otherwise.

Example:

```bash
flux run -n 2 python - <<'PY'
from mpi4py import MPI
import h5py
import heffte
comm = MPI.COMM_WORLD
print(f"rank {comm.rank}: h5py_mpi={h5py.get_config().mpi}, heffte={heffte.__version__}")
PY
```

---

# Short build cheat sheet

| Task | Command core |
|---|---|
| create env | `conda create -p ... python=3.11` |
| activate env | `conda activate /usr/workspace/.../heffte-py-rzadams` |
| load modules | `PrgEnv-gnu`, `gcc/12.2.0`, `cmake/3.29.2`, `craype-x86-trento`, `cray-fftw/3.3.10.11` |
| build `mpi4py` | `MPICC=$(which cc) pip install --no-binary=mpi4py mpi4py` |
| build `h5py` | `CC=$(which cc) HDF5_MPI=ON HDF5_DIR=... pip install --no-binary=h5py --no-deps h5py` |
| build HeFFTe | `cmake ... -D FFTW_ROOT=/opt/cray/pe/fftw/3.3.10.11/x86_milan ...` |

If you want, I can turn this into either:

| Option | Output |
|---|---|
| 1 | a polished `env_rzadams.sh` |
| 2 | a one-time bootstrap script that rebuilds everything |
| 3 | a `flux run` test script for your repo |

#!/bin/bash
module --force purge
module load PrgEnv-gnu/8.7.0
module load gcc/12.2.0
module load cmake/3.29.2
module load craype-x86-trento
module load cray-fftw/3.3.10.11

source /collab/usr/gapps/python/toss_4_x86_64_ib/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate /usr/workspace/zendejas/turbulence_post_process/.conda/envs/heffte-py-rzadams

export HDF5_MPI=ON
export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/cray/20.0
export PMI_LIBDIR=/opt/cray/pe/lib64
export CCE_LIBDIR=/opt/cray/pe/lib64/cce
export PYTHONPATH=/usr/workspace/zendejas/turbulence_post_process/third_party/heffte/install/share/heffte/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/workspace/zendejas/turbulence_post_process/third_party/heffte/install/lib64:/usr/workspace/zendejas/turbulence_post_process/third_party/heffte/install/lib:${CCE_LIBDIR}:${PMI_LIBDIR}:${HDF5_DIR}/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
