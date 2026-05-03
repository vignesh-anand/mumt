#!/usr/bin/env bash
# M3b asset bake: convert HSSD scene 102344049 into Unity-friendly geometry.
#
# Prereqs (one-time, both done via scripts/04_install_vr_assets.sh):
#   1) .venv-magnum exists with habitat_dataset_processing installed.
#   2) third_party/magnum-tools/linux-x64/python/ exists (manual artifact
#      download from github.com/mosra/magnum-ci).
#
# Output: _data_processing_output/data/scene_datasets/hssd-hab/...
# Next step: in Unity Editor, run Tools > Update Data Folder... and paste the
# absolute path printed at the end of this script.
#
# Run-time: 5-15 min for one HSSD scene. Decimation of objects is the bulk.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d .venv-magnum ]; then
    echo "ERROR: .venv-magnum not found. Run scripts/04_install_vr_assets.sh first." >&2
    exit 1
fi

if [ ! -d third_party/magnum-tools/linux-x64/python ]; then
    echo "ERROR: magnum-tools not extracted. See scripts/04_install_vr_assets.sh" >&2
    echo "       for the manual GitHub Actions download instructions." >&2
    exit 1
fi

OUT_DIR="$REPO_ROOT/_data_processing_output"

echo "==> processing HSSD scene 102344049 -> $OUT_DIR"
echo "    (this will take 5-15 min; object decimation is the bulk)"

.venv-magnum/bin/python scripts/process_unity_hssd.py \
    --input "$REPO_ROOT/data" \
    --output "$OUT_DIR"

PROCESSED_DATA="$OUT_DIR/data"
if [ ! -d "$PROCESSED_DATA" ]; then
    echo "ERROR: expected output at $PROCESSED_DATA but nothing was produced." >&2
    exit 1
fi

cat <<EOF

==========================================================================
Asset bake done. Output: $PROCESSED_DATA

Next step (Unity Editor, manual):
  1) Open the siro_hitl_unity_client project (already open in Editor).
  2) Tools > Update Data Folder...
  3) External Data Path:
        $PROCESSED_DATA
  4) Click "Update Data Folder". Wait for the AssetDatabase reimport
     (5-15 min, with intermittent freezes - that's normal).
  5) Validate by re-running scripts/06_run_sim_viewer_server.sh and hitting
     Play in Unity. The Game view should now show the FULL HSSD scene
     instead of empty geometry.
==========================================================================
EOF
