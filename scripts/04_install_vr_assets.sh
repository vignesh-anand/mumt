#!/usr/bin/env bash
# M3a.2: install Unity-client + magnum-tools asset-processing prerequisites.
#
# Idempotent. Re-run after each manual step. The script will detect what is
# already done and only do the missing pieces.
#
# 1) Clone siro_hitl_unity_client into third_party/.
# 2) Create .venv-magnum (a separate Python 3.10 venv) and install
#    habitat_dataset_processing + pycollada into it. Magnum's Python bindings
#    intentionally live outside this venv (see step 3).
# 3) If the user has manually extracted magnum-tools binaries to
#    third_party/magnum-tools/linux-x64/, write a magnum.pth so the venv can
#    find them. Otherwise print the manual download instructions and stop.
# 4) Validate the magnum bindings by importing the modules our processor needs.
#
# Usage: bash scripts/04_install_vr_assets.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=python3.10
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: $PY not found. The system Python 3.10 is required." >&2
    exit 1
fi

echo "==> ensuring third_party/siro_hitl_unity_client is present"
if [ ! -d third_party/siro_hitl_unity_client ]; then
    git clone --depth 1 \
        https://github.com/eundersander/siro_hitl_unity_client.git \
        third_party/siro_hitl_unity_client
else
    echo "    already cloned"
fi

echo "==> creating .venv-magnum (Python 3.10) for the asset processor"
if [ ! -d .venv-magnum ]; then
    "$PY" -m venv .venv-magnum
    .venv-magnum/bin/pip install --upgrade pip wheel >/dev/null
fi
echo "    venv ready: $(.venv-magnum/bin/python --version)"

echo "==> installing habitat_dataset_processing into .venv-magnum"
.venv-magnum/bin/pip install -q -e \
    third_party/habitat-lab/scripts/habitat_dataset_processing
.venv-magnum/bin/pip install -q pycollada

MAGNUM_DIR="$REPO_ROOT/third_party/magnum-tools/linux-x64"
PY_SP=$(.venv-magnum/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

if [ -d "$MAGNUM_DIR/python" ]; then
    echo "==> linking magnum bindings into .venv-magnum"
    echo "$MAGNUM_DIR/python" > "$PY_SP/magnum.pth"
    echo "    wrote $PY_SP/magnum.pth"

    echo "==> validating magnum import"
    if .venv-magnum/bin/python -c \
        "from magnum import math, meshtools, scenetools, trade" 2>&1; then
        echo "    OK"
    else
        echo "ERROR: magnum import failed. Re-check that the artifact you" >&2
        echo "       extracted matches your Python version (3.10) and arch." >&2
        exit 1
    fi
    echo
    echo "==> M3a.2 prerequisites done. Next:"
    echo "    bash scripts/05_process_unity_data.sh"
else
    cat <<'EOF'

==========================================================================
MANUAL STEP REQUIRED: download magnum-tools binaries.

Magnum doesn't ship a public direct URL; the binaries live as GitHub
Actions artifacts. You need a (free) GitHub login. Then:

  1. Open https://github.com/mosra/magnum-ci/actions/workflows/magnum-tools.yml
  2. Click the latest GREEN workflow run.
  3. Scroll to "Artifacts" at the bottom.
  4. Download "magnum-tools-vYYYY.MM-NNN-linux-x64" (Linux x86_64 build).
  5. Extract the zip so that this path exists:
         third_party/magnum-tools/linux-x64/python/magnum/__init__.py
     Example:
         mkdir -p third_party/magnum-tools
         cd third_party/magnum-tools
         unzip ~/Downloads/magnum-tools-*-linux-x64.zip
  6. Re-run this script: bash scripts/04_install_vr_assets.sh
==========================================================================
EOF
    exit 0
fi
