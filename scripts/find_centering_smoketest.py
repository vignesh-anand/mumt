#!/usr/bin/env python
"""Ground-truth check for the FindLabelController centering math.

Spawns a single Spot at a known pose, drops a humanoid at a known
*world* offset, renders the spot's head-cam frame, asks the YOLOE
server for ``human``, and prints:

- where the bbox actually appears in the image (left vs right)
- the world yaw needed to face the human (from the planted XZ)
- the world yaw the FindLabelController centering math predicts
  given the bbox center, the snap pose, and the camera HFOV.

If the centering math is right, the two yaws agree. Run this any time
the camera basis or yaw convention changes.

Note: this script does NOT touch the navmesh; it places the spot and
the human at hard-coded world positions so the sign check stays
independent of the scene's geometry.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import habitat_sim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mumt_sim.agent.detection import YoloeClient
from mumt_sim.agents import add_kinematic_humanoid, add_kinematic_spot
from mumt_sim.pan_tilt import PanTiltHead
from mumt_sim.scene import (
    SPOT_0_HEAD_AGENT_ID,
    head_sensor_uuids,
    make_sim,
)


HSSD_ROOT = REPO_ROOT / "data" / "scene_datasets" / "hssd-hab"
SCENE_DATASET_CONFIG = HSSD_ROOT / "hssd-hab.scene_dataset_config.json"
SCENE_INSTANCE = HSSD_ROOT / "scenes" / "102344049.scene_instance.json"

LIVE_HW = (480, 640)
SPOT_HEAD_HFOV_DEG = 110.0


def _heading_to(src_xz, dst_xz) -> float:
    """Same convention as mumt_sim.agent.tools._heading_to."""
    dx = dst_xz[0] - src_xz[0]
    dz = dst_xz[1] - src_xz[1]
    return math.atan2(-dz, dx)


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_predict_from_bbox(
    bbox_cx: float, img_w: int, snap_yaw: float, hfov_rad: float, sign: int
) -> float:
    """Run the centering math with an explicit sign so we can A/B test
    both conventions."""
    u_norm = (bbox_cx - 0.5 * img_w) / (0.5 * img_w)
    u_norm = max(-1.0, min(1.0, u_norm))
    half_hfov = 0.5 * hfov_rad
    cam_offset_rad = math.atan(u_norm * math.tan(half_hfov))
    return snap_yaw + sign * cam_offset_rad


def _sample_depth_at_pixel(depth: np.ndarray, cx: float, cy: float, win: int = 5):
    h, w = depth.shape[:2]
    px = max(0, min(w - 1, int(round(cx))))
    py = max(0, min(h - 1, int(round(cy))))
    y0, y1 = max(0, py - win), min(h, py + win + 1)
    x0, x1 = max(0, px - win), min(w, px + win + 1)
    patch = depth[y0:y1, x0:x1]
    finite = patch[np.isfinite(patch) & (patch > 0)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def main() -> None:
    os.chdir(REPO_ROOT)

    print(f"==> loading scene {SCENE_INSTANCE.name}", flush=True)
    sim = make_sim(
        scene_id=str(SCENE_INSTANCE),
        scene_dataset_config_file=str(SCENE_DATASET_CONFIG),
        image_hw=LIVE_HW,
    )
    if not sim.pathfinder.is_loaded:
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius = 0.3
        ns.agent_height = 1.5
        ns.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, ns)

    # Snap a navigable point to anchor the spot. We deliberately place
    # the human at +X, -Z relative to the spot (forward + image-right
    # if our coordinate trace is right).
    sample = sim.pathfinder.get_random_navigable_point()
    spot_xyz = np.array([float(sample[0]), float(sample[1]), float(sample[2])])

    # Try a few human placements until we find one that's both navigable
    # and visible from the spot's head cam (no walls between them).
    placements_world = [
        (+1.5,  0.0),  # forward only      -> bbox center
        (+1.5, -1.0),  # forward + (-Z)    -> if image-right == world -Z, RIGHT
        (+1.5, +1.0),  # forward + (+Z)    -> opposite
        (+2.5, -1.0),
        (+2.5, +1.0),
    ]

    s_rgb_uuid, s_depth_uuid = head_sensor_uuids(0)

    yoloe = YoloeClient(
        base_url=os.environ.get("MUMT_YOLOE_URL"),
        timeout_s=10.0,
    )
    print(f"==> YOLOE health = {yoloe.health().get('open', {}).get('backend')}",
          flush=True)

    out_dir = REPO_ROOT / "outputs" / "find_smoketest"
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, (dx, dz) in enumerate(placements_world):
        print()
        print(f"=== placement {k+1}: human at body-local (+X={dx:+.1f}, "
              f"+Z={dz:+.1f}) from spot ===")

        # Re-create the scene each time would be cleanest; for a quick
        # smoketest we rebuild only the kinematic spot/human via
        # remove + re-add.
        # Simplest: reset existing objects.
        rom = sim.get_rigid_object_manager()
        aom = sim.get_articulated_object_manager()
        for h in list(rom.get_object_handles()):
            rom.remove_object_by_handle(h)
        for h in list(aom.get_object_handles()):
            aom.remove_object_by_handle(h)

        spot_yaw = 0.0  # facing +X
        spot = add_kinematic_spot(sim, spot_xyz, yaw_rad=spot_yaw)

        # Place the human in absolute world coords. Since spot yaw=0,
        # body +X aligns with world +X.
        human_xyz = np.array([
            spot_xyz[0] + dx,
            spot_xyz[1],
            spot_xyz[2] + dz,
        ], dtype=np.float32)
        snap_h = sim.pathfinder.snap_point(human_xyz)
        human_xyz = np.asarray(snap_h, dtype=np.float32)
        if not np.all(np.isfinite(human_xyz)):
            print(f"  skip: snap_point returned NaN for "
                  f"({dx:+.1f}, {dz:+.1f})")
            continue
        add_kinematic_humanoid(sim, human_xyz, yaw_rad=math.pi, name="female_0")

        head = PanTiltHead(sim, agent_id=SPOT_0_HEAD_AGENT_ID, body_ao=spot)
        head.set_pan_tilt(pan=0.0, tilt=math.radians(10.0))
        head.sync()

        obs = sim.get_sensor_observations(SPOT_0_HEAD_AGENT_ID)
        rgb = obs[s_rgb_uuid][:, :, :3]
        depth = obs[s_depth_uuid]
        # Convert RGB->BGR for cv2/YOLOE.
        bgr = rgb[:, :, ::-1].copy()

        # Save the rendered frame so we can eyeball it.
        out_path = out_dir / f"placement_{k+1}_dx{dx:+.1f}_dz{dz:+.1f}.png"
        cv2.imwrite(str(out_path), bgr)

        H, W = rgb.shape[:2]
        snap_x = float(spot_xyz[0])
        snap_z = float(spot_xyz[2])
        gt_yaw = _heading_to((snap_x, snap_z),
                             (float(human_xyz[0]), float(human_xyz[2])))

        try:
            r = yoloe.detect_open(rgb, ["human"], conf=0.2,
                                  rgb_is_bgr=False)
        except Exception as exc:
            print(f"  YOLOE failed: {exc}")
            continue
        best = r.best_for_label("human")
        if best is None:
            print(f"  no 'human' detection (n_dets={len(r.detections)}) "
                  f"-- saved to {out_path.relative_to(REPO_ROOT)}")
            continue
        cx, cy = best.center_xy
        side = "RIGHT" if cx > 0.5 * W else ("LEFT" if cx < 0.5 * W else "CENTER")

        hfov = math.radians(SPOT_HEAD_HFOV_DEG)
        yaw_minus = _yaw_predict_from_bbox(cx, W, spot_yaw, hfov, sign=-1)
        yaw_plus = _yaw_predict_from_bbox(cx, W, spot_yaw, hfov, sign=+1)
        err_minus = math.degrees(_wrap_to_pi(yaw_minus - gt_yaw))
        err_plus = math.degrees(_wrap_to_pi(yaw_plus - gt_yaw))

        print(f"  bbox center px=({cx:.0f},{cy:.0f}) of {W}x{H} -> {side}  "
              f"conf={best.confidence:.2f}")
        print(f"  ground-truth yaw to face human       = "
              f"{math.degrees(gt_yaw):+7.2f} deg")
        print(f"  predicted (snap_yaw - cam_offset)    = "
              f"{math.degrees(yaw_minus):+7.2f} deg   err={err_minus:+6.2f} deg")
        print(f"  predicted (snap_yaw + cam_offset)    = "
              f"{math.degrees(yaw_plus):+7.2f} deg   err={err_plus:+6.2f} deg")
        winner = "minus" if abs(err_minus) < abs(err_plus) else "plus"
        print(f"  --> {winner.upper()} sign matches    "
              f"(saved render to {out_path.relative_to(REPO_ROOT)})")

        # ----- depth -> approach math -----
        z_cam = _sample_depth_at_pixel(depth, cx, cy)
        u_norm = (cx - 0.5 * W) / (0.5 * W)
        u_norm = max(-1.0, min(1.0, u_norm))
        cam_offset_rad = math.atan(u_norm * math.tan(0.5 * hfov))
        if z_cam is None:
            print("  depth: NO valid samples around bbox center")
            continue
        target_range_pred = z_cam / max(0.05, math.cos(cam_offset_rad))
        gt_horiz_range = math.hypot(
            float(human_xyz[0]) - snap_x,
            float(human_xyz[2]) - snap_z,
        )
        # Subtract a rough body-radius (~0.2 m) since depth bottoms out
        # on the human's surface, not their world-XZ centroid.
        print(f"  depth z_cam at bbox center = {z_cam:.2f} m")
        print(f"  predicted target horiz range = {target_range_pred:.2f} m")
        print(f"  ground-truth XZ range        = {gt_horiz_range:.2f} m  "
              f"(diff = {target_range_pred - gt_horiz_range:+.2f} m, mostly "
              f"= human body radius)")

        # Where we'd actually drive to land 1.0 m short of the target:
        target_yaw = yaw_minus
        forward = (math.cos(target_yaw), -math.sin(target_yaw))
        approach_d = 1.0
        drive_dist = max(0.0, target_range_pred - approach_d)
        approach_pt = (snap_x + drive_dist * forward[0],
                       snap_z + drive_dist * forward[1])
        target_pt = (snap_x + target_range_pred * forward[0],
                     snap_z + target_range_pred * forward[1])
        print(f"  estimated target world XZ = "
              f"({target_pt[0]:+.2f}, {target_pt[1]:+.2f})")
        print(f"  approach goto target XZ   = "
              f"({approach_pt[0]:+.2f}, {approach_pt[1]:+.2f})  "
              f"(spot drives {drive_dist:.2f} m)")

    sim.close()


if __name__ == "__main__":
    main()
