#!/usr/bin/env bash
# M3c: Run the custom mumt HITL server (scripts/mumt_hitl_app.py).
#
# Spawns 2 kinematic Spots + 1 humanoid in HSSD scene 102344049 and streams
# them to any connected siro_hitl_unity_client (Quest 2 APK + adb reverse, or
# Unity Editor on localhost). Right thumbstick teleops Spot 0 once the
# updated APK from M3c phase C is installed; until then only the head-pose
# embodiment of the humanoid is wired up.
#
# Usage:
#   bash scripts/07_run_mumt_hitl_server.sh
#
# Stops on Ctrl-C. Same headless / mock-handshake conventions as
# scripts/06_run_sim_viewer_server.sh; see README M3a section for context.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

# - habitat_hitl.experimental.headless.do_headless=True / window=null:
#       avoid the magnum.text.Renderer2D crash on this NVIDIA stack;
#       see README M3a.3 / "Problems and fixes".
# - +habitat_hitl.networking.mock_connection_params_dict.isClientReady=true:
#       protocol mismatch workaround for the older siro_hitl_unity_client
#       (HEAD = dbfa5a6) that doesn't send the isClientReady handshake field.
python scripts/mumt_hitl_app.py \
    habitat_hitl.networking.enable=True \
    habitat_hitl.networking.port=8888 \
    habitat_hitl.experimental.headless.do_headless=True \
    habitat_hitl.window=null \
    +habitat_hitl.networking.mock_connection_params_dict.isClientReady=true
