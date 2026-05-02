#!/usr/bin/env bash
# Bootstrap the isolated environment for the mumt project.
# - Creates .venv with Python 3.10
# - Fresh clones habitat-sim and habitat-lab into third_party/
# - Builds habitat-sim from source (with bullet)
# - pip installs habitat-lab + habitat-baselines + habitat-hitl
#
# Idempotent-ish: skips clones that already exist; re-runs pip installs cheaply.

set -euo pipefail

MUMT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${MUMT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"

echo "==> mumt root: ${MUMT_DIR}"
echo "==> python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version))"

# 1) venv
if [ ! -d ".venv" ]; then
    echo "==> creating venv at .venv"
    "${PYTHON_BIN}" -m venv .venv
else
    echo "==> .venv already exists; reusing"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# 2) clone third_party (fresh, isolated)
mkdir -p third_party
if [ ! -d "third_party/habitat-sim/.git" ]; then
    echo "==> cloning habitat-sim (with submodules)"
    git clone --recursive https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
else
    echo "==> third_party/habitat-sim already cloned; updating submodules"
    git -C third_party/habitat-sim submodule update --init --recursive
fi

if [ ! -d "third_party/habitat-lab/.git" ]; then
    echo "==> cloning habitat-lab"
    git clone --depth 1 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
else
    echo "==> third_party/habitat-lab already cloned"
fi

# 3) project Python deps (numpy, imageio, hf cli, hydra)
echo "==> installing project requirements"
pip install -r requirements.txt

# 4) build prereqs for habitat-sim (scikit-build-core, pybind11, cmake, ninja)
#    plus habitat-sim's own runtime requirements. We pre-install these so we can
#    pass --no-build-isolation in the next step (recommended by habitat-sim).
echo "==> installing habitat-sim build prereqs"
pip install "scikit-build-core>=0.10" "pybind11>=2.10" "cmake>=3.22" "ninja"
pip install -r third_party/habitat-sim/requirements.txt

# 5) build & install habitat-sim
# Cap C++ build parallelism to avoid OOM-killing on machines with limited RAM.
# Magnum's heavy templating means each cc1plus can transiently use 2-3 GB,
# so a default of 4 jobs is safe on a 16-32 GB machine. Override via env if you
# have plenty of RAM (e.g. MUMT_BUILD_JOBS=8 bash scripts/01_install_env.sh).
MUMT_BUILD_JOBS="${MUMT_BUILD_JOBS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="${MUMT_BUILD_JOBS}"
export MAKEFLAGS="-j${MUMT_BUILD_JOBS}"

echo "==> building habitat-sim (bullet, editable, no-build-isolation, jobs=${MUMT_BUILD_JOBS})"
pushd third_party/habitat-sim >/dev/null
export HABITAT_WITH_BULLET=ON
export HABITAT_BUILD_GUI_VIEWERS=ON
if command -v nvcc >/dev/null 2>&1; then
    echo "==> nvcc found, enabling CUDA"
    export HABITAT_WITH_CUDA=ON
else
    export HABITAT_WITH_CUDA=OFF
fi
pip install -e . --no-build-isolation -v
popd >/dev/null

# 6) habitat-lab + baselines + hitl (for milestones 2/3, not used in M1 script)
echo "==> installing habitat-lab stack"
pushd third_party/habitat-lab >/dev/null
pip install -e habitat-lab
pip install -e habitat-baselines
pip install -e habitat-hitl
popd >/dev/null

# 7) editable install of our own package
echo "==> installing mumt_sim (editable)"
pip install -e .

# 8) smoke test
echo "==> smoke test"
python -c "
import habitat_sim, habitat, habitat_baselines, habitat_hitl, mumt_sim
print('habitat_sim       :', habitat_sim.__version__)
print('habitat           :', habitat.__version__)
print('habitat_baselines :', habitat_baselines.__version__)
print('habitat_hitl      : ok')
print('mumt_sim          : ok')
"

echo "==> install_env complete"
