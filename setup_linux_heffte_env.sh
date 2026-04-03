#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-heffte-py-linux}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/environment.linux-mpi.yml}"
HEFFTE_DIR="${HEFFTE_DIR:-${PROJECT_ROOT}/third_party/heffte}"
HEFFTE_BUILD_DIR="${HEFFTE_BUILD_DIR:-${HEFFTE_DIR}/build}"
HEFFTE_INSTALL_DIR="${HEFFTE_INSTALL_DIR:-${HEFFTE_DIR}/install}"
HEFFTE_GIT_REF="${HEFFTE_GIT_REF:-}"
CONDA_EXE_DEFAULT="${HOME}/anaconda3/bin/conda"

if [[ -x "${CONDA_EXE_DEFAULT}" ]]; then
  CONDA_BASE="$("${CONDA_EXE_DEFAULT}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "Unable to find conda." >&2
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  conda env create -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

MPICC="${MPICC:-${CONDA_PREFIX}/bin/mpicc}"
MPICXX="${MPICXX:-${CONDA_PREFIX}/bin/mpicxx}"
MPIRUN="${MPIRUN:-${CONDA_PREFIX}/bin/mpirun}"

for tool in "${MPICC}" "${MPICXX}" "${MPIRUN}"; do
  if [[ ! -x "${tool}" ]]; then
    echo "Missing MPI tool: ${tool}" >&2
    exit 1
  fi
done

python - <<'PY'
import h5py
import mpi4py.MPI as MPI

print("h5py version =", h5py.__version__)
print("h5py mpi =", h5py.get_config().mpi)
print("MPI library version =", MPI.Get_library_version().splitlines()[0])

if not h5py.get_config().mpi:
    raise SystemExit("Expected an MPI-enabled h5py build.")
PY

mkdir -p "$(dirname "${HEFFTE_DIR}")"

if [[ ! -d "${HEFFTE_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/icl-utk-edu/heffte.git "${HEFFTE_DIR}"
elif [[ -n "${HEFFTE_GIT_REF}" ]]; then
  git -C "${HEFFTE_DIR}" fetch --depth 1 origin "${HEFFTE_GIT_REF}"
fi

if [[ -n "${HEFFTE_GIT_REF}" ]]; then
  git -C "${HEFFTE_DIR}" checkout --force "${HEFFTE_GIT_REF}"
fi

cmake -S "${HEFFTE_DIR}" -B "${HEFFTE_BUILD_DIR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_SHARED_LIBS=ON \
  -D CMAKE_INSTALL_PREFIX="${HEFFTE_INSTALL_DIR}" \
  -D CMAKE_C_COMPILER="${MPICC}" \
  -D CMAKE_CXX_COMPILER="${MPICXX}" \
  -D FFTW_ROOT="${CONDA_PREFIX}" \
  -D Heffte_ENABLE_FFTW=ON \
  -D Heffte_ENABLE_PYTHON=ON \
  -D Python_EXECUTABLE="${CONDA_PREFIX}/bin/python"

cmake --build "${HEFFTE_BUILD_DIR}" -j"$(nproc)"
cmake --install "${HEFFTE_BUILD_DIR}"

ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${ACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/heffte.sh" <<EOF
export PYTHONPATH="${HEFFTE_INSTALL_DIR}/share/heffte/python:\${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${HEFFTE_INSTALL_DIR}/lib64:${HEFFTE_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"
EOF

DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${DEACTIVATE_DIR}"
cat > "${DEACTIVATE_DIR}/heffte.sh" <<EOF
export PYTHONPATH=\$(python - <<'PY'
import os
entries = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p and p != "${HEFFTE_INSTALL_DIR}/share/heffte/python"]
print(":".join(entries))
PY
)
export LD_LIBRARY_PATH=\$(python - <<'PY'
import os
blocked = {"${HEFFTE_INSTALL_DIR}/lib64", "${HEFFTE_INSTALL_DIR}/lib"}
entries = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and p not in blocked]
print(":".join(entries))
PY
)
EOF

conda deactivate
conda activate "${ENV_NAME}"

python - <<'PY'
import h5py
import heffte

print("h5py mpi =", h5py.get_config().mpi)
print("heffte version =", heffte.__version__)
print("heffte FFTW enabled =", getattr(heffte.heffte_config, "enable_fftw", False))

if not h5py.get_config().mpi:
    raise SystemExit("Expected an MPI-enabled h5py build after activation.")
if not getattr(heffte.heffte_config, "enable_fftw", False):
    raise SystemExit("HeFFTe was built without FFTW support.")
PY

echo
echo "Environment ready."
echo "Activate with: conda activate ${ENV_NAME}"
echo "yt comes from: ${ENV_FILE}"
echo "Run spectra with: mpirun -n 4 python ComputeSpectra.py your_data.h5 --backend heffte_fftw --no-plot"
echo "Run converter with: mpirun -n 4 python tools/convert_txt_to_hdf5.py your_data.txt"
