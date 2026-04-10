#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --partition=pbatch
#SBATCH --time=23:59:00
#SBATCH --job-name=turbulence_postprocess
#SBATCH --output=turbulence_postprocess.%j.out
#SBATCH --error=turbulence_postprocess.%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/run_tuolumne_postprocess.sh <sampled_data_root> [<sampled_data_root> ...]

Example:
  sbatch scripts/run_tuolumne_postprocess.sh \
    /path/to/blast_tgv3Dk2r5_SampledData \
    /path/to/blast_tgv3Dk2r4_SampledData

Behavior:
  - uses one batch allocation total
  - finds cycle_* subdirectories under each provided root
  - runs one srun step per cycle inside that same allocation
  - appends density and pressure scalar TXT files when they are available
  - renders only:
      velocity_magnitude
      vorticity_magnitude
      density (if available)
      pressure (if available)

Notes:
  - if a velocity .h5 already exists in the cycle directory, it is used directly
  - otherwise the velocity .txt is used, and main.py may delete that .txt after
    a successful TXT -> HDF5 conversion
  - supports both legacy velocity_sampled_data_uniform_interpolated_cycle_* names
    and SampledData* names for velocity inputs
  - override partition/account/time at submit time as needed, for example:
      sbatch --partition=pdebug --time=00:30:00 scripts/run_tuolumne_postprocess.sh ...
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

POSTPROC_RANKS="${POSTPROC_RANKS:-${SLURM_NTASKS:-96}}"
POSTPROC_NODES="${POSTPROC_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
POSTPROC_CONDA_SH="${POSTPROC_CONDA_SH:-/g/g11/zendejas/anaconda3/etc/profile.d/conda.sh}"
POSTPROC_CONDA_ENV="${POSTPROC_CONDA_ENV:-heffte-py-tuo}"
POSTPROC_REPO="${POSTPROC_REPO:-$HOME/turbulence_post_process}"
POSTPROC_MAIN="${POSTPROC_MAIN:-$POSTPROC_REPO/main.py}"
POSTPROC_HEFFTE_PY="${POSTPROC_HEFFTE_PY:-/p/lustre5/zendejas/third_party/heffte/install/share/heffte/python}"
POSTPROC_HEFFTE_LIB="${POSTPROC_HEFFTE_LIB:-/p/lustre5/zendejas/third_party/heffte/install/lib64}"
POSTPROC_HDF5_DIR="${POSTPROC_HDF5_DIR:-/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2}"
POSTPROC_GCC_MODULE="${POSTPROC_GCC_MODULE:-gcc/13.3.1}"

module --force purge
module load StdEnv
module load craype-x86-trento
module load "${POSTPROC_GCC_MODULE}"
module load cray-hdf5-parallel/1.14.3.7
module load cray-fftw/3.3.10.11
module load rocm/6.4.0
module load cmake/3.29.2

set +u
export PS1="${PS1-}"
source "${POSTPROC_CONDA_SH}"
conda activate "${POSTPROC_CONDA_ENV}"
set -u

export HDF5_MPI=ON
export HDF5_DIR="${POSTPROC_HDF5_DIR}"
export PYTHONPATH="${POSTPROC_HEFFTE_PY}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${POSTPROC_HEFFTE_LIB}:/opt/cray/pe/lib64/cce:${HDF5_DIR}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1

first_matching_file() {
  local search_dir="$1"
  shift
  local pattern
  local match
  for pattern in "$@"; do
    match="$(find "${search_dir}" -maxdepth 1 -type f -name "${pattern}" | sort -V | head -n 1)"
    if [[ -n "${match}" ]]; then
      printf '%s\n' "${match}"
      return 0
    fi
  done
  return 1
}

python -c "import h5py, heffte; print('h5py mpi =', h5py.get_config().mpi); print('HeFFTe =', heffte.__version__)"

mapfile -t cycle_dirs < <(
  for root in "$@"; do
    find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'cycle_*'
  done | sort -V
)

if [[ ${#cycle_dirs[@]} -eq 0 ]]; then
  echo "No cycle_* directories found under the provided roots." >&2
  exit 1
fi

echo "Found ${#cycle_dirs[@]} cycle directories."
echo "Using ${POSTPROC_RANKS} ranks across ${POSTPROC_NODES} node(s)."

failures=0

for cycle_dir in "${cycle_dirs[@]}"; do
  velocity_txt="$(first_matching_file "${cycle_dir}" \
    'velocity_sampled_data_uniform_interpolated_cycle_*.txt' \
    'SampledData*.txt' || true)"
  velocity_h5="$(first_matching_file "${cycle_dir}" \
    'velocity_sampled_data_uniform_interpolated_cycle_*.h5' \
    'SampledData*.h5' || true)"
  density_txt="$(first_matching_file "${cycle_dir}" \
    'density_sampled_data_uniform_interpolated_cycle_*.txt' || true)"
  pressure_txt="$(first_matching_file "${cycle_dir}" \
    'pressure_sampled_data_uniform_interpolated_cycle_*.txt' || true)"

  echo
  echo "======================================================================"
  echo "Cycle directory: ${cycle_dir}"
  echo "======================================================================"

  if [[ -n "${velocity_h5}" ]]; then
    main_input="${velocity_h5}"
  elif [[ -n "${velocity_txt}" ]]; then
    main_input="${velocity_txt}"
    echo "Warning: using velocity TXT input. main.py may delete it after successful conversion."
  else
    echo "Skipping: no velocity sampled-data TXT or HDF5 found." >&2
    failures=$((failures + 1))
    continue
  fi

  echo "Main input:   ${main_input}"
  if [[ -n "${density_txt}" ]]; then
    echo "Density TXT:  ${density_txt}"
  else
    echo "Density TXT:  not found (skipping density field)"
  fi
  if [[ -n "${pressure_txt}" ]]; then
    echo "Pressure TXT: ${pressure_txt}"
  else
    echo "Pressure TXT: not found (skipping pressure field)"
  fi

  postproc_args=(
    "${main_input}"
    --slice-field velocity_magnitude
    --slice-field vorticity_magnitude
    --slice-format pdf
    --slice-dpi 600
  )
  if [[ -n "${density_txt}" ]]; then
    postproc_args+=(--scalar-file "${density_txt}" --slice-field density)
  fi
  if [[ -n "${pressure_txt}" ]]; then
    postproc_args+=(--scalar-file "${pressure_txt}" --slice-field pressure)
  fi

  if ! srun -n "${POSTPROC_RANKS}" -N "${POSTPROC_NODES}" python "${POSTPROC_MAIN}" "${postproc_args[@]}"; then
    echo "Post-processing failed for ${cycle_dir}" >&2
    failures=$((failures + 1))
  fi
done

echo
echo "======================================================================"
if [[ ${failures} -eq 0 ]]; then
  echo "Post-processing completed successfully for all cycle directories."
else
  echo "Post-processing completed with ${failures} failure(s)." >&2
fi
echo "======================================================================"

exit "${failures}"
