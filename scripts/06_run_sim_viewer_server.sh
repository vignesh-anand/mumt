#!/usr/bin/env bash
# M3a.3: run habitat-hitl's sim_viewer as a websocket server on ws://127.0.0.1:8888
# loading our M1 HSSD scene 102344049. The Unity client (siro_hitl_unity_client)
# defaults to that port, so once this is running you can press Play in the
# Unity editor and the scene should stream.
#
# Usage:
#   bash scripts/06_run_sim_viewer_server.sh
#
# Stops on Ctrl-C. The sim_viewer also opens a small desktop window; feel free
# to ignore it - the rendering you care about is in Unity.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

# We use the regular (non-articulated) HSSD config because that is what M1
# downloaded into data/. The articulated config + hssd-hab-articulated objects
# is a future M3c concern.
# NOTE: 'mock_connection_params_dict' carries a single key "isClientReady=true".
# This is a workaround for the eundersander/siro_hitl_unity_client (HEAD =
# dbfa5a6) being stuck on an older protocol where the client never sends an
# isClientReady handshake field but the server (interprocess_record.py:64)
# asserts it must exist. Without this override, the server silently drops the
# connection right after the handshake "Client is ready!" log line, and the
# Unity client falls into a 4s reconnect loop. See M3a section of README.
python third_party/habitat-lab/examples/hitl/sim_viewer/sim_viewer.py \
    habitat_hitl.networking.enable=True \
    habitat_hitl.networking.port=8888 \
    habitat_hitl.experimental.headless.do_headless=True \
    habitat_hitl.window=null \
    +habitat_hitl.networking.mock_connection_params_dict.isClientReady=true \
    sim_viewer.dataset=data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json \
    sim_viewer.scene=102344049.scene_instance.json
