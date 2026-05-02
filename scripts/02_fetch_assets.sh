#!/usr/bin/env bash
# Fetch all assets needed for milestone 1 into ./data:
# - Spot URDF + meshes (hab_spot_arm)
# - Humanoid URDFs (humanoid_data, SMPL-X)
# - One HSSD scene (102344049) plus its referenced stages/objects (gated, requires HF auth).
#
# Run after 01_install_env.sh.

set -euo pipefail

MUMT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${MUMT_DIR}"

# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p data

# 1) HF login (skip if already logged in)
if hf auth whoami >/dev/null 2>&1; then
    echo "==> HF already authenticated as: $(hf auth whoami)"
else
    echo "==> Authenticate with Hugging Face."
    echo "    You must first accept the dataset terms at:"
    echo "      https://huggingface.co/datasets/hssd/hssd-hab"
    echo "    Then create a read token at:"
    echo "      https://huggingface.co/settings/tokens"
    hf auth login
fi

# 2) Spot URDF + humanoid URDFs via the official habitat-sim downloader.
echo "==> downloading hab_spot_arm + habitat_humanoids into data/"
python -m habitat_sim.utils.datasets_download \
    --uids hab_spot_arm habitat_humanoids \
    --data-path data/

# 3) One HSSD scene. We pull the dataset config + stages + the chosen scene
#    instance + objects referenced in that scene only.
HSSD_SCENE="102344049"
echo "==> downloading HSSD scene ${HSSD_SCENE} into data/scene_datasets/hssd-hab/"
hf download hssd/hssd-hab --repo-type dataset \
    --local-dir data/scene_datasets/hssd-hab \
    --include "hssd-hab.scene_dataset_config.json" \
    --include "stages/**" \
    --include "scenes/${HSSD_SCENE}*" \
    --include "scenes-uncluttered/${HSSD_SCENE}*" \
    --include "objects/**"

echo "==> sanity listing:"
echo "    Spot URDF :"
ls -1 data/robots/hab_spot_arm 2>/dev/null | head -5 || echo "    (missing)"
echo "    Humanoid  :"
ls -1 data/humanoids/humanoid_data 2>/dev/null | head -5 || echo "    (missing)"
echo "    HSSD scene:"
ls -1 data/scene_datasets/hssd-hab/scenes 2>/dev/null | head -5 || echo "    (missing)"

echo "==> fetch_assets complete"
