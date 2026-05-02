#!/usr/bin/env python
"""M2a deliverable: keyboard teleop of Spot 0 in HSSD scene 102344049.

Reuses the M1 scene + triangle layout (human + Spot 1 frozen in place), then
opens a split-screen pygame window: Spot 0's head camera on the left, the
observer bird's-eye on the right. Drive Spot 0 with WASD (W/S linear, A/D yaw),
arrows for pan (left/right) and tilt (up/down). Hold shift for a 2x speed
boost. R resets Spot 0 to the start pose. Esc / window close quits.

The Spots and the human were spawned *after* the navmesh was baked, so they
do not act as obstacles for Spot 0. Walls and stage geometry do (via
``pathfinder.try_step`` sliding).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

import habitat_sim
import magnum as mn

from mumt_sim.agents import add_kinematic_humanoid, add_kinematic_spot
from mumt_sim.display import InputState, SplitScreenWindow
from mumt_sim.pan_tilt import PanTiltHead
from mumt_sim.scene import (
    OBSERVER_AGENT_ID,
    SPOT_0_HEAD_AGENT_ID,
    SPOT_1_HEAD_AGENT_ID,
    head_sensor_uuids,
    make_sim,
)
from mumt_sim.spawn import equilateral_triangle_around, find_open_spawn_spot
from mumt_sim.teleop import SpotTeleop, TeleopInput, TeleopParams


REPO_ROOT = Path(__file__).resolve().parents[1]

HSSD_ROOT = REPO_ROOT / "data" / "scene_datasets" / "hssd-hab"
SCENE_DATASET_CONFIG = HSSD_ROOT / "hssd-hab.scene_dataset_config.json"
SCENE_INSTANCE = HSSD_ROOT / "scenes" / "102344049.scene_instance.json"

# Live render is half the M1 offline resolution so we hit 60 fps comfortably.
LIVE_HW = (480, 640)
TARGET_FPS = 60

INITIAL_SPOT_PAN = 0.0
INITIAL_SPOT_TILT = math.radians(10.0)


def _resolve_scene_paths() -> tuple[str, str]:
    if not SCENE_DATASET_CONFIG.exists():
        sys.exit(
            f"ERROR: scene dataset config not found at {SCENE_DATASET_CONFIG}.\n"
            f"Run scripts/02_fetch_assets.sh first."
        )
    if not SCENE_INSTANCE.exists():
        sys.exit(
            f"ERROR: scene instance not found at {SCENE_INSTANCE}.\n"
            f"Re-run scripts/02_fetch_assets.sh; the include patterns may need widening."
        )
    return str(SCENE_INSTANCE), str(SCENE_DATASET_CONFIG)


def _yaw_facing(src_xyz, target_xz) -> float:
    dx = float(target_xz[0] - src_xyz[0])
    dz = float(target_xz[1] - src_xyz[2])
    return math.atan2(-dz, dx)


def _build_input(state: InputState) -> TeleopInput:
    """Translate a window ``InputState`` into the framework-agnostic ``TeleopInput``."""
    return TeleopInput(
        forward=state.forward,
        backward=state.backward,
        yaw_left=state.yaw_left,
        yaw_right=state.yaw_right,
        pan_left=state.pan_left,
        pan_right=state.pan_right,
        tilt_up=state.tilt_up,
        tilt_down=state.tilt_down,
        boost=state.boost,
        reset=state.reset_pressed,
    )


def main() -> None:
    os.chdir(REPO_ROOT)
    scene_id, scene_dataset_cfg = _resolve_scene_paths()

    print(f"==> loading scene {scene_id}")
    sim = make_sim(
        scene_id=scene_id,
        scene_dataset_config_file=scene_dataset_cfg,
        image_hw=LIVE_HW,
    )

    if not sim.pathfinder.is_loaded:
        print("==> recomputing navmesh (HSSD scene has none cached)")
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = 0.3
        navmesh_settings.agent_height = 1.5
        navmesh_settings.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    print("==> finding open spot on navmesh")
    TRIANGLE_SIDE = 1.0
    triangle_radius = TRIANGLE_SIDE / math.sqrt(3.0)
    min_clearance = triangle_radius + 0.7
    center, clearance = find_open_spawn_spot(
        sim.pathfinder, min_clearance=min_clearance, n_samples=1500
    )
    print(
        f"    center at {tuple(round(float(x), 3) for x in center)} "
        f"(clearance {clearance:.2f} m)"
    )

    # M1-style layout: order is human, spot_0, spot_1.
    triangle_rotation = float(np.random.default_rng().uniform(0, 2 * math.pi))
    p_human, p_spot_0, p_spot_1 = equilateral_triangle_around(
        center, radius=triangle_radius, rotation=triangle_rotation
    )
    centroid_xz = (float(center[0]), float(center[2]))

    print("==> spawning kinematic agents (each facing center)")
    spot_0 = add_kinematic_spot(sim, p_spot_0, yaw_rad=_yaw_facing(p_spot_0, centroid_xz))
    spot_1 = add_kinematic_spot(sim, p_spot_1, yaw_rad=_yaw_facing(p_spot_1, centroid_xz))
    add_kinematic_humanoid(
        sim, p_human, yaw_rad=_yaw_facing(p_human, centroid_xz), name="female_0"
    )

    head_0 = PanTiltHead(sim, agent_id=SPOT_0_HEAD_AGENT_ID, body_ao=spot_0)
    head_1 = PanTiltHead(sim, agent_id=SPOT_1_HEAD_AGENT_ID, body_ao=spot_1)
    head_0.set_pan_tilt(pan=INITIAL_SPOT_PAN, tilt=INITIAL_SPOT_TILT)
    head_1.set_pan_tilt(pan=INITIAL_SPOT_PAN, tilt=INITIAL_SPOT_TILT)
    head_0.sync()
    head_1.sync()  # Spot 1 stays put; sync once.

    # Observer at angled bird's-eye, looking at the cluster's initial centroid.
    print("==> placing observer cam (angled bird's eye)")
    centroid = np.mean(np.stack([p_spot_0, p_spot_1, p_human]), axis=0)
    spot_axis = np.array(p_spot_1)[[0, 2]] - np.array(p_spot_0)[[0, 2]]
    if np.linalg.norm(spot_axis) < 1e-3:
        spot_axis = np.array([1.0, 0.0])
    spot_axis = spot_axis / np.linalg.norm(spot_axis)
    perp = np.array([-spot_axis[1], spot_axis[0]])

    OBSERVER_RADIAL = 2.5
    OBSERVER_HEIGHT = 5.0
    observer_pos = np.array([
        centroid[0] + float(perp[0]) * OBSERVER_RADIAL,
        centroid[1] + OBSERVER_HEIGHT,
        centroid[2] + float(perp[1]) * OBSERVER_RADIAL,
    ])
    target = mn.Vector3(float(centroid[0]), float(centroid[1]) + 0.7, float(centroid[2]))
    eye = mn.Vector3(*observer_pos)
    look_at_m = mn.Matrix4.look_at(eye, target, mn.Vector3.y_axis())
    o_q = mn.Quaternion.from_matrix(look_at_m.rotation_normalized())

    observer = sim.get_agent(OBSERVER_AGENT_ID)
    o_state = observer.get_state()
    o_state.position = [float(x) for x in observer_pos]
    o_state.rotation = [o_q.vector.x, o_q.vector.y, o_q.vector.z, o_q.scalar]
    observer.set_state(o_state)

    teleop = SpotTeleop(sim, body_ao=spot_0, head=head_0, params=TeleopParams())

    s0_rgb_uuid, _ = head_sensor_uuids(0)

    print("==> opening teleop window. WASD to drive, arrows pan/tilt, R reset, Esc quit.")
    window = SplitScreenWindow(
        left_hw=LIVE_HW, right_hw=LIVE_HW, title="mumt M2a - teleop spot 0"
    )

    try:
        # Prime the window with one frame BEFORE polling so cv2 has a real
        # surface to receive focus / key events on.
        obs = sim.get_sensor_observations([OBSERVER_AGENT_ID, SPOT_0_HEAD_AGENT_ID])
        window.show(
            obs[SPOT_0_HEAD_AGENT_ID][s0_rgb_uuid][:, :, :3],
            obs[OBSERVER_AGENT_ID]["observer_rgb"][:, :, :3],
            hud_lines=["loading..."],
        )

        while not window.should_close():
            dt = window.tick(TARGET_FPS)
            state = window.poll_input()
            if state.quit_pressed:
                break

            controls = _build_input(state)
            teleop.step(dt, controls)

            obs = sim.get_sensor_observations(
                [OBSERVER_AGENT_ID, SPOT_0_HEAD_AGENT_ID]
            )
            left = obs[SPOT_0_HEAD_AGENT_ID][s0_rgb_uuid][:, :, :3]
            right = obs[OBSERVER_AGENT_ID]["observer_rgb"][:, :, :3]

            pos = teleop.state.position
            hud = [
                f"pos ({pos.x:+.2f}, {pos.y:+.2f}, {pos.z:+.2f}) m",
                f"yaw {math.degrees(teleop.state.yaw):+6.1f} deg   "
                f"pan {math.degrees(teleop.state.pan):+6.1f} deg   "
                f"tilt {math.degrees(teleop.state.tilt):+6.1f} deg",
                f"{1.0/dt:5.1f} fps   {'BOOST' if controls.boost else '     '}",
            ]
            window.show(left, right, hud_lines=hud)
    finally:
        window.close()
        sim.close()


if __name__ == "__main__":
    main()
