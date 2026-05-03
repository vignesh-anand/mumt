#!/usr/bin/env python
"""M-Agent.2 Slices A + B: 2-spot teleop + live top-down coverage map.

Layout:
  +--------------------+--------------------+
  | Spot 0 head RGB    | Spot 1 head RGB    |   top row
  | (360 x 480)        | (360 x 480)        |
  +--------------------+--------------------+
  |      TOP-DOWN COVERAGE (live)           |   bottom row, full width
  |      (360 x 960, fine grid resampled)   |
  +-----------------------------------------+

Total window: 720 tall x 960 wide. Sized to fit comfortably on a HiDPI
laptop (e.g. 2880x1800 @ 2x => 1440x900 logical) without overflowing.

Controls (both Spots driven simultaneously):
  Spot 0 (left pane)        Spot 1 (right pane)
  --------------------      --------------------
  W           forward       arrow-up    forward
  S           backward      arrow-down  backward
  A           yaw left      arrow-left  yaw left
  D           yaw right     arrow-right yaw right

  Shift                     2x speed boost (applies to both)
  R                         reset both Spots to start pose
  Esc                       quit

Coverage (slice B): a 10 cm fine grid spans the navmesh AABB. Each tick we
back-project both Spots' depth pixels into world XZ and stamp every cell
each pixel hits. Spot 0 paints in cyan, Spot 1 in magenta; intersection
shades toward white. Cells fade from full-bright (just-seen) to ~30%
brightness (5+ minutes old). Non-navigable cells are nearly black.

Heads stay locked at the initial slight downward tilt for now; head sweep
is driven by the body's yaw. Head pan/tilt controls can be added back in
a follow-up if needed.
"""
from __future__ import annotations

import faulthandler
import math
import os
import sys
from pathlib import Path

faulthandler.enable()

import cv2
import numpy as np

# cv2/habitat_sim GLX conflict workaround: bind cv2's GTK/GLX backend to the
# X display BEFORE importing habitat_sim. Otherwise habitat_sim's EGL/GL
# context binds first and the next cv2.namedWindow call segfaults silently
# on NVIDIA drivers. We start a window thread + open + paint + pump a
# throwaway window, then destroy it; the X11/GTK side stays initialised.
cv2.startWindowThread()
_warmup = "_mumt_cv2_warmup"
cv2.namedWindow(_warmup, cv2.WINDOW_NORMAL)
cv2.imshow(_warmup, np.zeros((64, 64, 3), dtype=np.uint8))
cv2.waitKey(1)
cv2.destroyWindow(_warmup)
cv2.waitKey(1)

import habitat_sim

from mumt_sim.agent.coverage import CoverageMap, CoverageMapConfig
from mumt_sim.agents import add_kinematic_humanoid, add_kinematic_spot
from mumt_sim.display import InputState, MultiPaneWindow
from mumt_sim.pan_tilt import PanTiltHead
from mumt_sim.scene import (
    SPOT_0_HEAD_AGENT_ID,
    SPOT_1_HEAD_AGENT_ID,
    head_sensor_uuids,
    make_sim,
)
from mumt_sim.spawn import (
    navmesh_path_midpoint,
    sample_far_pair_navmesh,
)
from mumt_sim.teleop import SpotTeleop, TeleopInput, TeleopParams


REPO_ROOT = Path(__file__).resolve().parents[1]

HSSD_ROOT = REPO_ROOT / "data" / "scene_datasets" / "hssd-hab"
SCENE_DATASET_CONFIG = HSSD_ROOT / "hssd-hab.scene_dataset_config.json"
SCENE_INSTANCE = HSSD_ROOT / "scenes" / "102344049.scene_instance.json"

# Sensor render resolution (kept at the M2a default; pane displays downscale).
LIVE_HW = (480, 640)

# Display pane sizes (tuned for ~1440x900 logical laptop displays).
HEAD_PANE_HW = (360, 480)
COVERAGE_PANE_HW = (360, 960)

TARGET_FPS = 60

INITIAL_SPOT_PAN = 0.0
INITIAL_SPOT_TILT = math.radians(10.0)

# Spot head camera horizontal FOV. Must match make_sim's spot_head_hfov_deg
# default (mumt_sim/scene.make_sim) so depth back-projection lines up.
SPOT_HEAD_HFOV_DEG = 110.0

# Per-spot coverage tints in BGR (cv2's channel order).
SPOT_COVERAGE_BGR = [
    (255, 255, 0),   # Spot 0 -> cyan
    (255,   0, 255), # Spot 1 -> magenta
]


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


def _wasd_to_input(state: InputState) -> TeleopInput:
    """WASD body driving. Shift = boost. R = reset."""
    return TeleopInput(
        forward=state.forward,
        backward=state.backward,
        yaw_left=state.yaw_left,
        yaw_right=state.yaw_right,
        boost=state.boost,
        reset=state.reset_pressed,
    )


def _arrows_to_input(state: InputState) -> TeleopInput:
    """Arrow-key body driving. We re-purpose ``InputState``'s pan/tilt fields
    (which are bound to the arrow keys) as body axes, since head pan/tilt
    is unused in this slice.

    arrow-up    -> tilt_up    -> forward
    arrow-down  -> tilt_down  -> backward
    arrow-left  -> pan_left   -> yaw left
    arrow-right -> pan_right  -> yaw right
    """
    return TeleopInput(
        forward=state.tilt_up,
        backward=state.tilt_down,
        yaw_left=state.pan_left,
        yaw_right=state.pan_right,
        boost=state.boost,
        reset=state.reset_pressed,
    )


def main() -> None:
    os.chdir(REPO_ROOT)
    scene_id, scene_dataset_cfg = _resolve_scene_paths()

    print(f"==> loading scene {scene_id}", flush=True)
    sim = make_sim(
        scene_id=scene_id,
        scene_dataset_config_file=scene_dataset_cfg,
        image_hw=LIVE_HW,
    )

    if not sim.pathfinder.is_loaded:
        print("==> recomputing navmesh (HSSD scene has none cached)", flush=True)
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        # Habitat default (0.30 m). Pairs with ``SPOT_HEAD_OFFSET.x == 0.20 m``
        # in mumt_sim/agents.py: the head sits ~10 cm inside the body's
        # wall-clearance disk, so it can't poke through walls and stamp
        # phantom coverage on the far side, while the body can still squeeze
        # through 60 cm passages.
        navmesh_settings.agent_radius = 0.3
        navmesh_settings.agent_height = 1.5
        navmesh_settings.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    print("==> picking far-apart spawn pair on navmesh", flush=True)
    p_spot_0, p_spot_1, geo_d = sample_far_pair_navmesh(
        sim.pathfinder,
        n_samples=600,
        min_clearance=0.7,
        refine_iters=3,
    )
    print(
        f"    spot_0 at {tuple(round(float(x), 3) for x in p_spot_0)}",
        flush=True,
    )
    print(
        f"    spot_1 at {tuple(round(float(x), 3) for x in p_spot_1)}",
        flush=True,
    )
    print(f"    geodesic distance between spots = {geo_d:.2f} m", flush=True)

    p_human = navmesh_path_midpoint(sim.pathfinder, p_spot_0, p_spot_1)
    print(
        f"    human at path midpoint {tuple(round(float(x), 3) for x in p_human)}",
        flush=True,
    )

    print("==> spawning kinematic agents (spots face each other)", flush=True)
    p1_xz = (float(p_spot_1[0]), float(p_spot_1[2]))
    p0_xz = (float(p_spot_0[0]), float(p_spot_0[2]))
    spot_0 = add_kinematic_spot(sim, p_spot_0, yaw_rad=_yaw_facing(p_spot_0, p1_xz))
    spot_1 = add_kinematic_spot(sim, p_spot_1, yaw_rad=_yaw_facing(p_spot_1, p0_xz))
    add_kinematic_humanoid(
        sim, p_human, yaw_rad=_yaw_facing(p_human, p1_xz), name="female_0"
    )

    head_0 = PanTiltHead(sim, agent_id=SPOT_0_HEAD_AGENT_ID, body_ao=spot_0)
    head_1 = PanTiltHead(sim, agent_id=SPOT_1_HEAD_AGENT_ID, body_ao=spot_1)
    head_0.set_pan_tilt(pan=INITIAL_SPOT_PAN, tilt=INITIAL_SPOT_TILT)
    head_1.set_pan_tilt(pan=INITIAL_SPOT_PAN, tilt=INITIAL_SPOT_TILT)
    head_0.sync()
    head_1.sync()

    teleops = [
        SpotTeleop(sim, body_ao=spot_0, head=head_0, params=TeleopParams()),
        SpotTeleop(sim, body_ao=spot_1, head=head_1, params=TeleopParams()),
    ]

    s0_rgb_uuid, s0_depth_uuid = head_sensor_uuids(0)
    s1_rgb_uuid, s1_depth_uuid = head_sensor_uuids(1)
    spot_bodies = [spot_0, spot_1]
    head_agent_ids = [SPOT_0_HEAD_AGENT_ID, SPOT_1_HEAD_AGENT_ID]
    depth_uuids = [s0_depth_uuid, s1_depth_uuid]

    print("==> building coverage map (sampling navigability per fine cell)",
          flush=True)
    coverage = CoverageMap(
        sim=sim,
        n_spots=2,
        config=CoverageMapConfig(
            fine_cell_m=0.10,
            max_range_m=5.0,
            pixel_stride=4,
        ),
    )
    n_nav = int(coverage.is_navigable.sum())
    n_total = coverage.nz * coverage.nx
    print(
        f"    grid {coverage.nz} x {coverage.nx} cells "
        f"({coverage.x_max - coverage.x_min:.1f} x "
        f"{coverage.z_max - coverage.z_min:.1f} m AABB)",
        flush=True,
    )
    print(
        f"    floor probe Y = {coverage.y_probe:+.3f} m, "
        f"navigable = {n_nav}/{n_total} cells ({100.0*n_nav/n_total:.1f}%)",
        flush=True,
    )

    print("==> opening teleop window. WASD drives Spot 0, arrows drive Spot 1.",
          flush=True)
    window = MultiPaneWindow(
        pane_grid=[
            [HEAD_PANE_HW, HEAD_PANE_HW],
            [COVERAGE_PANE_HW],
        ],
        title="mumt M-Agent.2 - 2-spot teleop + live coverage",
    )

    sim_t = 0.0  # accumulated sim time (seconds)

    # Spot 0 -> WASD, Spot 1 -> arrows (per user-preferred mapping).
    input_for_spot = (_wasd_to_input, _arrows_to_input)
    label_for_spot = ("Spot 0 (WASD)", "Spot 1 (arrows)")

    pane_h, pane_w = COVERAGE_PANE_HW

    def render_coverage_pane() -> np.ndarray:
        """Render the coverage map at fine resolution, then aspect-
        preserving INTER_NEAREST upscale to fill the pane (letterboxed).
        Coarse 5-m grid lines + chess labels and Spot markers are drawn
        AFTER the upscale at the matching cell-pixel scale so they stay
        crisp and proportional."""
        fine = coverage.render_topdown(
            t_now=sim_t, spot_colors_bgr=SPOT_COVERAGE_BGR
        )
        nz, nx = fine.shape[:2]
        scale = min(pane_h / nz, pane_w / nx)
        new_h = max(1, int(round(nz * scale)))
        new_w = max(1, int(round(nx * scale)))
        upscaled = cv2.resize(fine, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
        y0 = (pane_h - new_h) // 2
        x0 = (pane_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = upscaled

        spot_poses = []
        for i in (0, 1):
            p = spot_bodies[i].translation
            spot_poses.append((float(p.x), float(p.z), teleops[i].state.yaw))
        view = canvas[y0:y0 + new_h, x0:x0 + new_w]
        coverage.draw_coarse_grid(view, cell_pixel_scale=scale)
        coverage.draw_spot_markers(
            view, spot_poses=spot_poses, spot_colors_bgr=SPOT_COVERAGE_BGR,
            cell_pixel_scale=scale,
        )
        return canvas

    try:
        # Prime the window with one frame BEFORE polling so cv2 has a real
        # surface to receive focus / key events on.
        sensor_ids = head_agent_ids
        obs = sim.get_sensor_observations(sensor_ids)
        spot0_frame = obs[SPOT_0_HEAD_AGENT_ID][s0_rgb_uuid][:, :, :3]
        spot1_frame = obs[SPOT_1_HEAD_AGENT_ID][s1_rgb_uuid][:, :, :3]
        window.show([
            (spot0_frame, [label_for_spot[0], "loading..."]),
            (spot1_frame, [label_for_spot[1], "loading..."]),
            (render_coverage_pane(), [], True),
        ])

        while not window.should_close():
            dt = window.tick(TARGET_FPS)
            state = window.poll_input()
            if state.quit_pressed:
                break
            sim_t += dt

            teleops[0].step(dt, input_for_spot[0](state))
            teleops[1].step(dt, input_for_spot[1](state))

            obs = sim.get_sensor_observations(sensor_ids)
            spot0_frame = obs[SPOT_0_HEAD_AGENT_ID][s0_rgb_uuid][:, :, :3]
            spot1_frame = obs[SPOT_1_HEAD_AGENT_ID][s1_rgb_uuid][:, :, :3]

            # Coverage update: per-Spot depth back-projection + self-cell stamp.
            cells_stamped = [0, 0]
            for spot_id in (0, 1):
                depth = obs[head_agent_ids[spot_id]][depth_uuids[spot_id]]
                cam_T_world = CoverageMap.head_camera_world_transform(
                    sim, agent_id=head_agent_ids[spot_id],
                    sensor_uuid=depth_uuids[spot_id],
                )
                body_pos = spot_bodies[spot_id].translation
                cells_stamped[spot_id] = coverage.update_from_depth(
                    spot_id=spot_id,
                    t_now=sim_t,
                    cam_T_world=cam_T_world,
                    depth=depth,
                    hfov_deg=SPOT_HEAD_HFOV_DEG,
                    body_xz=(float(body_pos.x), float(body_pos.z)),
                )
                coverage.stamp_self_cell(
                    spot_id=spot_id,
                    t_now=sim_t,
                    world_xyz=(float(body_pos.x), float(body_pos.y), float(body_pos.z)),
                )

            coverage_img = render_coverage_pane()

            huds = []
            for i, t in enumerate(teleops):
                pos = t.state.position
                huds.append([
                    label_for_spot[i],
                    f"pos ({pos.x:+.2f}, {pos.y:+.2f}, {pos.z:+.2f}) m",
                    f"yaw {math.degrees(t.state.yaw):+6.1f} deg",
                ])
            huds[0].append(f"{1.0/dt:5.1f} fps   {'BOOST' if state.boost else '     '}")

            cov_seen_total = int((coverage.last_seen_t.max(axis=-1) > -1e8).sum())
            cov_navigable = int(coverage.is_navigable.sum())
            cov_pct = (100.0 * cov_seen_total / max(1, cov_navigable))
            label_s0 = coverage.coarse_label_for_world_xz(
                float(spot_bodies[0].translation.x),
                float(spot_bodies[0].translation.z),
            ) or "--"
            label_s1 = coverage.coarse_label_for_world_xz(
                float(spot_bodies[1].translation.x),
                float(spot_bodies[1].translation.z),
            ) or "--"
            cov_hud = [
                f"coverage map (live, {coverage.cfg.coarse_cell_m:g} m sectors)",
                f"t={sim_t:6.1f}s   "
                f"seen {cov_seen_total}/{cov_navigable} cells ({cov_pct:4.1f}%)",
                f"s0 sector {label_s0}   s1 sector {label_s1}",
                f"stamped this tick: spot0={cells_stamped[0]}  spot1={cells_stamped[1]}",
            ]

            window.show([
                (spot0_frame, huds[0]),
                (spot1_frame, huds[1]),
                (coverage_img, cov_hud, True),
            ])
    finally:
        window.close()
        sim.close()


if __name__ == "__main__":
    main()
