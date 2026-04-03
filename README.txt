Python Scripts Setup and Run Guide
==================================

Scripts covered:
  main.py
  tools/ComputeSpectra.py
  tools/convert_txt_to_hdf5.py
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
  mpirun -n 4 python tools/ComputeSpectra.py your_data.h5 --backend heffte_fftw --no-plot
  mpirun -n 4 python tools/convert_txt_to_hdf5.py your_data.txt
  mpirun -n 4 python tools/visualize_velocity_yt.py your_data.h5 --slice z:center --field vx

Integrated pipeline:

  mpirun -n 4 python main.py your_data.txt

What main.py now does:
  1. Accepts .txt or .h5 input.
  2. Converts TXT to structured FFT-ready HDF5 when needed.
  3. Optionally appends one or more scalar sampled-data TXT files into /fields/<scalar_name>
     of the structured HDF5.
  4. Runs the FFT/spectra workflow on the resulting HDF5.
  5. Writes one or more slice PDFs after the FFT step.

Files written by the integrated pipeline:
  - the structured .h5 is written next to the original .txt input
  - the FFT spectra .txt and spectra metadata .txt are written next to that .h5 file
  - the slice plots are written under slice_plots/ next to that .h5 file
  - the raw slice data are written under slice_data/ as one combined <base>_slices.h5 file

Default slice outputs from main.py:
  - xy_center_velocity_magnitude.pdf
  - xy_face_velocity_magnitude.pdf
  - yz_face_velocity_magnitude.pdf
  - zx_face_velocity_magnitude.pdf
  - xy_center_vorticity_magnitude.pdf
  - xy_face_vorticity_magnitude.pdf
  - yz_face_vorticity_magnitude.pdf
  - zx_face_vorticity_magnitude.pdf
  - one combined slice_data/<base>_slices.h5 file containing all saved slice arrays

Slice output defaults:
  - format: pdf
  - save dpi: 600
  - square figure size: 8.0 inches

Canonical slice field names:
  - velocity_magnitude
  - vorticity_magnitude
  - vx, vy, vz
  - wx, wy, wz

The combined slice HDF5 stores:
  - all saved 2D slice arrays
  - horizontal and vertical coordinates for each slice
  - axis, plane index, plane coordinate, step, time, and source-file metadata

This lets you replot slices later without recomputing the FFT/vorticity workflow.

Optional scalar field inputs:
  - pass one or more sampled-data scalar TXT files to main.py with:
      --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt
      --scalar-file pressure_sampled_data_uniform_interpolated_cycle_0.txt
  - the scalar dataset names are parsed from the "Sampled Data, <name>" header line
  - each scalar is appended to /fields/<scalar_name> in the structured HDF5
  - the same default slices are then written for velocity, vorticity, and all appended scalars

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
      --slice-field density
  - tools/visualize_velocity_yt.py:
      --format {pdf,png}
      --dpi 600
      --figsize 10

Output:

  data/slice_plots/SampledData0_xy_center_velocity_magnitude.pdf
  data/slice_plots/SampledData0_xy_center_vorticity_magnitude.pdf
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

Use the standalone reader/replot tool to inspect what is available:

  python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 --list

Example replot:

  python tools/replot_slice_data.py data/slice_data/SampledData0_slices.h5 \
    --field velocity_magnitude \
    --slice xy_center \
    --cmap viridis

Example scalar workflow:

  mpirun -n 4 python main.py velocity_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file density_sampled_data_uniform_interpolated_cycle_0.txt \
    --scalar-file pressure_sampled_data_uniform_interpolated_cycle_0.txt

Useful controls:
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

  module purge
  module load craype-x86-trento
  module load gcc/13.3.1-magic
  module load cray-mpich/9.1.0
  module load cray-hdf5-parallel
  module load cray-fftw/3.3.10.11
  module load rocm/6.4.0
  module load cmake/3.29.2


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

Use the GNU HDF5 variant, not the Cray one:

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

  module purge
  module load craype-x86-trento
  module load gcc/13.3.1-magic
  module load cray-mpich/9.1.0
  module load cray-hdf5-parallel
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

  module purge
  module load craype-x86-trento
  module load gcc/13.3.1-magic
  module load cray-mpich/9.1.0
  module load cray-hdf5-parallel
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
