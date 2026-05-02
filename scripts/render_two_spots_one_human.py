#!/usr/bin/env python
"""M1 deliverable: load HSSD scene 102344049, find an open spot on the navmesh,
arrange two kinematic Spots and one kinematic humanoid in a 1 m equilateral
triangle all facing the centre, then render observer + each Spot's head and
write 5 PNGs to renders/.

Both Spots' heads are tilted up ~10 deg so the cameras frame the other agents'
torsos instead of the floor.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import habitat_sim
import magnum as mn

from mumt_sim.agents import add_kinematic_humanoid, add_kinematic_spot
from mumt_sim.pan_tilt import PanTiltHead
from mumt_sim.scene import (
    OBSERVER_AGENT_ID,
    SPOT_0_HEAD_AGENT_ID,
    SPOT_1_HEAD_AGENT_ID,
    head_sensor_uuids,
    make_sim,
)
from mumt_sim.spawn import equilateral_triangle_around, find_open_spawn_spot


REPO_ROOT = Path(__file__).resolve().parents[1]

HSSD_ROOT = REPO_ROOT / "data" / "scene_datasets" / "hssd-hab"
SCENE_DATASET_CONFIG = HSSD_ROOT / "hssd-hab.scene_dataset_config.json"
SCENE_INSTANCE = HSSD_ROOT / "scenes" / "102344049.scene_instance.json"

RENDERS_DIR = REPO_ROOT / "renders"
IMAGE_HW = (720, 1280)

# Both Spots stare straight ahead so their head cams catch the other two
# triangle members. A small upward tilt centers the FOV on the human's torso
# instead of the floor since the head camera sits at ~0.7 m and the human's
# centre of mass at ~1.0 m.
SPOT_HEAD_PAN = 0.0
SPOT_HEAD_TILT = math.radians(10.0)


def _resolve_scene_paths() -> tuple[str, str]:
    """Resolve the scene files; fail loudly if the asset fetch step was skipped."""
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


def _depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    """Convert metres -> millimetres clipped to uint16 range. Saved as PNG-16."""
    mm = np.clip(depth * 1000.0, 0, 65535)
    return mm.astype(np.uint16)


def main() -> None:
    os.chdir(REPO_ROOT)  # all asset paths in mumt_sim are relative to repo root.
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    scene_id, scene_dataset_cfg = _resolve_scene_paths()

    print(f"==> loading scene {scene_id}")
    sim = make_sim(
        scene_id=scene_id,
        scene_dataset_config_file=scene_dataset_cfg,
        image_hw=IMAGE_HW,
    )

    # HSSD scenes don't ship with a pre-baked navmesh. Compute one before
    # sampling, and crucially include the placed static objects (cars,
    # furniture, etc.) as obstacles - otherwise the "open spot" finder will
    # happily pick a navigable patch with a car parked through the middle.
    if not sim.pathfinder.is_loaded:
        print("==> recomputing navmesh (HSSD scene has none cached)")
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = 0.3
        navmesh_settings.agent_height = 1.5
        navmesh_settings.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    print("==> finding open spot on navmesh")
    # We need a circular open area large enough for the agents to stand in a
    # triangle (~1m vertex-to-vertex) plus a little margin for the human's
    # ~0.7m T-pose arm reach.
    TRIANGLE_SIDE = 1.0
    triangle_radius = TRIANGLE_SIDE / math.sqrt(3.0)  # ~0.577 m
    min_clearance = triangle_radius + 0.7
    center, clearance = find_open_spawn_spot(
        sim.pathfinder, min_clearance=min_clearance, n_samples=1500
    )
    print(f"    center at {tuple(round(float(x), 3) for x in center)} "
          f"(clearance {clearance:.2f} m)")

    # Equilateral triangle around the center; pick a random rotation so we
    # don't always end up with the same orientation. Order: human, spot_0, spot_1.
    triangle_rotation = float(np.random.default_rng().uniform(0, 2 * math.pi))
    p_human, p_spot_0, p_spot_1 = equilateral_triangle_around(
        center, radius=triangle_radius, rotation=triangle_rotation
    )
    centroid_xz = (float(center[0]), float(center[2]))

    def _yaw_facing(src_xyz, target_xz) -> float:
        dx = float(target_xz[0] - src_xyz[0])
        dz = float(target_xz[1] - src_xyz[2])
        return math.atan2(-dz, dx)

    print("==> spawning kinematic agents (each facing center)")
    spot_0 = add_kinematic_spot(sim, p_spot_0, yaw_rad=_yaw_facing(p_spot_0, centroid_xz))
    spot_1 = add_kinematic_spot(sim, p_spot_1, yaw_rad=_yaw_facing(p_spot_1, centroid_xz))
    add_kinematic_humanoid(
        sim, p_human, yaw_rad=_yaw_facing(p_human, centroid_xz), name="female_0"
    )

    # Bind pan/tilt heads to each Spot.
    head_0 = PanTiltHead(sim, agent_id=SPOT_0_HEAD_AGENT_ID, body_ao=spot_0)
    head_1 = PanTiltHead(sim, agent_id=SPOT_1_HEAD_AGENT_ID, body_ao=spot_1)
    head_0.set_pan_tilt(pan=SPOT_HEAD_PAN, tilt=SPOT_HEAD_TILT)
    head_1.set_pan_tilt(pan=SPOT_HEAD_PAN, tilt=SPOT_HEAD_TILT)
    head_0.sync()
    head_1.sync()

    # Angled bird's-eye over the cluster: height clears typical interior walls,
    # tilt down at ~55 deg so the agents read as standing rather than dots.
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
    print(f"    observer at {tuple(round(float(x), 3) for x in observer_pos)}")

    # Render. Pass a list of agent_ids so we get a {agent_id: {sensor: obs}} dict;
    # the default (single int) only returns observations for agent 0.
    print("==> rendering")
    obs = sim.get_sensor_observations(
        [OBSERVER_AGENT_ID, SPOT_0_HEAD_AGENT_ID, SPOT_1_HEAD_AGENT_ID]
    )

    out_observer = RENDERS_DIR / "observer.png"
    out_s0_rgb = RENDERS_DIR / "spot_0_head_rgb.png"
    out_s0_depth = RENDERS_DIR / "spot_0_head_depth.png"
    out_s1_rgb = RENDERS_DIR / "spot_1_head_rgb.png"
    out_s1_depth = RENDERS_DIR / "spot_1_head_depth.png"

    # When the simulator has multiple agents the observation dict is keyed by
    # agent_id at the top level. Each Spot's head sensors have unique uuids
    # (see mumt_sim.scene.head_sensor_uuids) so they don't collide.
    s0_rgb_uuid, s0_depth_uuid = head_sensor_uuids(0)
    s1_rgb_uuid, s1_depth_uuid = head_sensor_uuids(1)
    imageio.imwrite(out_observer, obs[OBSERVER_AGENT_ID]["observer_rgb"][:, :, :3])
    imageio.imwrite(out_s0_rgb, obs[SPOT_0_HEAD_AGENT_ID][s0_rgb_uuid][:, :, :3])
    imageio.imwrite(out_s0_depth, _depth_to_uint16(obs[SPOT_0_HEAD_AGENT_ID][s0_depth_uuid]))
    imageio.imwrite(out_s1_rgb, obs[SPOT_1_HEAD_AGENT_ID][s1_rgb_uuid][:, :, :3])
    imageio.imwrite(out_s1_depth, _depth_to_uint16(obs[SPOT_1_HEAD_AGENT_ID][s1_depth_uuid]))

    print(f"==> wrote {out_observer}")
    print(f"==> wrote {out_s0_rgb}")
    print(f"==> wrote {out_s0_depth}")
    print(f"==> wrote {out_s1_rgb}")
    print(f"==> wrote {out_s1_depth}")

    sim.close()


if __name__ == "__main__":
    main()
