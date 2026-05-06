#!/usr/bin/env python
"""M-Agent.3 Slice B: two-Spot chat harness with an LLM orchestrator.

Layout (vertical stack so it fits in a ~960 x 1800 right-of-screen
column on a 2880 x 1800 laptop):

  +-----------------------------------+
  |   Spot 0 head POV  + HUD overlay  |
  +-----------------------------------+
  |   Spot 1 head POV  + HUD overlay  |
  +-----------------------------------+
  |                                   |
  |  Shared coverage map (5 m sectors)|
  |                                   |
  +-----------------------------------+

User chats in the terminal. The orchestrator is an LLM whose only
job is to route each user message to spot 0, spot 1, or both. Spot
agents then run their own ReAct loop and stream replies back to the
terminal verbatim.

    you> spot 0 search sector b2, spot 1 search d2
    [orch] -> spot[0]: search sector b2
    [orch] -> spot[1]: search d2
    spot0> Searching B2.
    spot0> [tool] search(sector='B2')
    spot1> Searching D2.
    spot1> [tool] search(sector='D2')
    ...

Special chat commands:
    :abort      tell BOTH agents to stop their running primitive
    :abort 0    tell only spot 0 to stop
    :quit       clean shutdown
"""
from __future__ import annotations

import faulthandler
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

faulthandler.enable()

import cv2
import numpy as np

# cv2 / habitat_sim GLX init order: open + close a throwaway cv2 window
# BEFORE habitat_sim is imported (same dance as the other scripts).
cv2.startWindowThread()
_warmup = "_mumt_cv2_warmup"
cv2.namedWindow(_warmup, cv2.WINDOW_NORMAL)
cv2.imshow(_warmup, np.zeros((64, 64, 3), dtype=np.uint8))
cv2.waitKey(1)
cv2.destroyWindow(_warmup)
cv2.waitKey(1)

import habitat_sim

from mumt_sim.agent.coverage import CoverageMap, CoverageMapConfig
from mumt_sim.agent.detection import OnDemandDetector, YoloeClient
from mumt_sim.agent.loop import (
    AgentClient,
    AgentLoop,
    EventBus,
    StdinChatReader,
    ToolDispatcher,
)
from mumt_sim.agent.memory import MemoryTable, default_jsonl_path
from mumt_sim.agent.orchestrator import OrchestratorClient, OrchestratorLoop
from mumt_sim.agent.perception import (
    CaptionWorker,
    GeminiClient,
    OnDemandCaptioner,
)
from mumt_sim.agent.recall import OnDemandRecaller, RecallClient
from mumt_sim.agent.tools import Controller, ControllerCtx, PrimitiveResult
from mumt_sim.agents import add_kinematic_humanoid, add_kinematic_spot
from mumt_sim.display import MultiPaneWindow
from mumt_sim.pan_tilt import PanTiltHead
from mumt_sim.scene import (
    SPOT_0_HEAD_AGENT_ID,
    SPOT_1_HEAD_AGENT_ID,
    head_sensor_uuids,
    make_sim,
)
from mumt_sim.spawn import navmesh_path_midpoint, sample_far_pair_navmesh
from mumt_sim.teleop import SpotTeleop, TeleopParams


REPO_ROOT = Path(__file__).resolve().parents[1]
HSSD_ROOT = REPO_ROOT / "data" / "scene_datasets" / "hssd-hab"
SCENE_DATASET_CONFIG = HSSD_ROOT / "hssd-hab.scene_dataset_config.json"
SCENE_INSTANCE = HSSD_ROOT / "scenes" / "102344049.scene_instance.json"

LIVE_HW = (480, 640)

# Vertical stack sized in LOGICAL pixels for a 2x HiDPI Ubuntu desktop:
# the cv2 canvas reports e.g. 480 px wide but the WM upscales 2x so
# it occupies ~960 physical px of the right-screen column. Habitat
# still renders at 480x640 and we let cv2 downscale into the small
# panes so HiDPI text + sensor frames stay sharp on the display.
POV_PANE_HW = (270, 480)
COVERAGE_PANE_HW = (360, 480)

TARGET_FPS = 60
SPOT_HEAD_HFOV_DEG = 110.0
INITIAL_SPOT_PAN = 0.0
INITIAL_SPOT_TILT = math.radians(10.0)

# Per-spot tints for the coverage pane.
SPOT_COVERAGE_BGR = [
    (255, 255, 0),  # cyan  -> spot 0
    (0, 255, 255),  # yellow -> spot 1
]

NUM_SPOTS = 2


def _resolve_scene_paths() -> tuple[str, str]:
    if not SCENE_DATASET_CONFIG.exists():
        sys.exit(
            f"ERROR: scene dataset config not found at {SCENE_DATASET_CONFIG}.\n"
            f"Run scripts/02_fetch_assets.sh first."
        )
    if not SCENE_INSTANCE.exists():
        sys.exit(
            f"ERROR: scene instance not found at {SCENE_INSTANCE}.\n"
            f"Re-run scripts/02_fetch_assets.sh."
        )
    return str(SCENE_INSTANCE), str(SCENE_DATASET_CONFIG)


def _yaw_facing(src_xyz, target_xz) -> float:
    dx = float(target_xz[0] - src_xyz[0])
    dz = float(target_xz[1] - src_xyz[2])
    return math.atan2(-dz, dx)


def _wrap(text: str, width: int) -> list[str]:
    text = (text or "").strip().replace("\n", " ")
    if not text:
        return []
    out: list[str] = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        out.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        out.append(text)
    return out


def _print_primitive_result(
    spot_id: int, name: Optional[str], result: PrimitiveResult,
) -> None:
    fx, fz, fyaw = result.final_pose
    name_disp = name or result.primitive
    print(
        f"\n[primitive] spot{spot_id} {name_disp} -> {result.status} "
        f"({result.reason}) t={result.t_elapsed_s:.1f}s "
        f"pose=({fx:+.2f},{fz:+.2f}, yaw {math.degrees(fyaw):+5.1f}deg)",
        flush=True,
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
        print("==> recomputing navmesh", flush=True)
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius = 0.3
        ns.agent_height = 1.5
        ns.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, ns)

    print("==> picking far-apart spawn pair on navmesh", flush=True)
    p_spot_0, p_spot_1, geo_d = sample_far_pair_navmesh(
        sim.pathfinder, n_samples=600, min_clearance=0.7, refine_iters=3,
    )
    p_human = navmesh_path_midpoint(sim.pathfinder, p_spot_0, p_spot_1)
    print(
        f"    spot_0 {tuple(round(float(x), 3) for x in p_spot_0)}\n"
        f"    spot_1 {tuple(round(float(x), 3) for x in p_spot_1)}\n"
        f"    human  {tuple(round(float(x), 3) for x in p_human)}\n"
        f"    geodesic gap {geo_d:.2f}m",
        flush=True,
    )

    p1_xz = (float(p_spot_1[0]), float(p_spot_1[2]))
    p0_xz = (float(p_spot_0[0]), float(p_spot_0[2]))
    spot_0 = add_kinematic_spot(
        sim, p_spot_0, yaw_rad=_yaw_facing(p_spot_0, p1_xz),
    )
    spot_1 = add_kinematic_spot(
        sim, p_spot_1, yaw_rad=_yaw_facing(p_spot_1, p0_xz),
    )
    human_ao = add_kinematic_humanoid(
        sim, p_human, yaw_rad=_yaw_facing(p_human, p1_xz), name="female_0",
    )

    head_0 = PanTiltHead(sim, agent_id=SPOT_0_HEAD_AGENT_ID, body_ao=spot_0)
    head_1 = PanTiltHead(sim, agent_id=SPOT_1_HEAD_AGENT_ID, body_ao=spot_1)
    for h in (head_0, head_1):
        h.set_pan_tilt(pan=INITIAL_SPOT_PAN, tilt=INITIAL_SPOT_TILT)
        h.sync()

    teleops = [
        SpotTeleop(sim, body_ao=spot_0, head=head_0, params=TeleopParams()),
        SpotTeleop(sim, body_ao=spot_1, head=head_1, params=TeleopParams()),
    ]
    spot_bodies = [spot_0, spot_1]
    head_agent_ids = [SPOT_0_HEAD_AGENT_ID, SPOT_1_HEAD_AGENT_ID]
    rgb_uuids: list = []
    depth_uuids: list = []
    for sid in range(NUM_SPOTS):
        rgb_u, depth_u = head_sensor_uuids(sid)
        rgb_uuids.append(rgb_u)
        depth_uuids.append(depth_u)

    coverage = CoverageMap(
        sim=sim, n_spots=NUM_SPOTS,
        config=CoverageMapConfig(
            fine_cell_m=0.10, max_range_m=5.0, pixel_stride=4,
        ),
    )
    n_nav = int(coverage.is_navigable.sum())
    print(
        f"    coverage grid {coverage.nz}x{coverage.nx} cells, "
        f"navigable {n_nav}",
        flush=True,
    )

    # ----- ambient captioner + on-demand pools (shared) -----------------
    memory_path = default_jsonl_path()
    memory = MemoryTable(jsonl_path=memory_path)
    print(f"==> memory table -> {memory_path}", flush=True)

    caption_workers: List[CaptionWorker] = []
    on_demand_captioner: Optional[OnDemandCaptioner] = None
    try:
        caption_client = GeminiClient(
            model=os.environ.get(
                "MUMT_CAPTION_MODEL", "gemini-3.1-flash-lite-preview",
            ),
        )
        for sid in range(NUM_SPOTS):
            cw = CaptionWorker(
                spot_id=sid, client=caption_client, memory=memory,
                period_s=2.0,
            )
            cw.start()
            caption_workers.append(cw)
        on_demand_captioner = OnDemandCaptioner(caption_client, max_workers=6)
        print(
            f"==> captioner running for {NUM_SPOTS} spots "
            f"(model={caption_client.model})",
            flush=True,
        )
    except Exception as exc:
        print(
            f"==> WARNING: captioning disabled ({exc}). "
            f"Set GEMINI_API_KEY to enable.",
            flush=True,
        )

    on_demand_detector: Optional[OnDemandDetector] = None
    try:
        yc = YoloeClient(
            base_url=os.environ.get("MUMT_YOLOE_URL"), timeout_s=4.0,
        )
        try:
            health = yc.health()
            backend = (health.get("open") or {}).get("backend", "?")
            print(
                f"==> YOLOE detector ready ({yc.base_url}, backend={backend})",
                flush=True,
            )
        except Exception as ping_exc:
            print(
                f"==> WARNING: YOLOE health probe failed at {yc.base_url} "
                f"({ping_exc}). find() will surface per-call errors.",
                flush=True,
            )
        on_demand_detector = OnDemandDetector(yc, max_workers=4)
    except Exception as exc:
        print(f"==> WARNING: YOLOE detector disabled ({exc})", flush=True)

    on_demand_recaller: Optional[OnDemandRecaller] = None
    try:
        rc = RecallClient(
            model=os.environ.get(
                "MUMT_RECALL_MODEL", "gemini-3.1-flash-lite-preview",
            ),
        )
        on_demand_recaller = OnDemandRecaller(rc, max_workers=2)
        print(f"==> recall LLM ready (model={rc.model})", flush=True)
    except Exception as exc:
        print(
            f"==> WARNING: recall LLM disabled ({exc}). "
            f"Set GEMINI_API_KEY to enable.",
            flush=True,
        )

    # ----- per-spot agent loops + dispatcher -----------------------------
    try:
        agent_client = AgentClient(
            model=os.environ.get(
                "MUMT_AGENT_MODEL", "gemini-3.1-flash-lite-preview",
            ),
        )
        print(f"==> agent LLM ready (model={agent_client.model})", flush=True)
    except Exception as exc:
        sys.exit(
            f"ERROR: agent client failed to initialise ({exc}). "
            f"Set GEMINI_API_KEY before launching."
        )

    dispatcher = ToolDispatcher()

    # Per-spot state snapshot (read by each AgentLoop).
    state_lock = threading.Lock()
    state_snapshots: List[Dict[str, Any]] = [
        {
            "pose_xz": (0.0, 0.0),
            "yaw_rad": 0.0,
            "sector": None,
            "sim_t": 0.0,
            "coverage_summary": "",
            "running_tool": None,
            "user_pose_xz": None,
            "user_sector": None,
        }
        for _ in range(NUM_SPOTS)
    ]

    def make_get_state(sid: int):
        def _get_state() -> Dict[str, Any]:
            with state_lock:
                return dict(state_snapshots[sid])
        return _get_state

    # Per-spot HUD trace.
    last_speak: List[Dict[str, Optional[Any]]] = [
        {"text": None, "t": None} for _ in range(NUM_SPOTS)
    ]
    last_thinking: List[Dict[str, Optional[Any]]] = [
        {"text": None, "t": None} for _ in range(NUM_SPOTS)
    ]
    last_action: List[Dict[str, Optional[Any]]] = [
        {"text": None, "t": None} for _ in range(NUM_SPOTS)
    ]

    def on_speak(spot_id: int, text: str) -> None:
        print(f"\nspot{spot_id}> {text}", flush=True)
        with state_lock:
            last_speak[spot_id]["text"] = text
            last_speak[spot_id]["t"] = time.monotonic()

    def on_thinking(spot_id: int, text: str) -> None:
        with state_lock:
            last_thinking[spot_id]["text"] = text
            last_thinking[spot_id]["t"] = time.monotonic()

    def on_action(spot_id: int, text: str) -> None:
        print(f"\nspot{spot_id}> [tool] {text}", flush=True)
        with state_lock:
            last_action[spot_id]["text"] = text
            last_action[spot_id]["t"] = time.monotonic()

    last_alert: List[Dict[str, Optional[Any]]] = [
        {"text": None, "t": None} for _ in range(NUM_SPOTS)
    ]

    def on_alert(spot_id: int, description: str) -> None:
        bar = "!" * 60
        sys.stdout.write("\a")  # terminal bell
        print(
            f"\n{bar}\n"
            f"!!! ALERT from spot{spot_id} !!!\n"
            f"!!! {description}\n"
            f"{bar}",
            flush=True,
        )
        with state_lock:
            last_alert[spot_id]["text"] = description
            last_alert[spot_id]["t"] = time.monotonic()

    spot_buses: List[EventBus] = []
    agents: List[AgentLoop] = []
    for sid in range(NUM_SPOTS):
        bus = EventBus(maxsize=128)
        agent = AgentLoop(
            spot_id=sid,
            client=agent_client,
            dispatcher=dispatcher,
            bus=bus,
            coverage=coverage,
            get_state=make_get_state(sid),
            on_demand_captioner=on_demand_captioner,
            on_demand_detector=on_demand_detector,
            on_demand_recaller=on_demand_recaller,
            on_speak=on_speak,
            on_thinking=on_thinking,
            on_action=on_action,
            on_alert=on_alert,
        )
        agent.start()
        spot_buses.append(bus)
        agents.append(agent)

    # ----- orchestrator --------------------------------------------------
    try:
        orch_client = OrchestratorClient(
            num_spots=NUM_SPOTS,
            model=os.environ.get(
                "MUMT_ORCH_MODEL", "gemini-3.1-flash-lite-preview",
            ),
        )
        print(f"==> orchestrator LLM ready (model={orch_client.model})", flush=True)
    except Exception as exc:
        sys.exit(
            f"ERROR: orchestrator client failed to initialise ({exc}). "
            f"Set GEMINI_API_KEY before launching."
        )

    orch_bus = EventBus(maxsize=64)

    def on_route(spot_ids: Sequence[int], message: str) -> None:
        ids_str = ",".join(str(i) for i in spot_ids)
        print(f"\n[orch] -> spot[{ids_str}]: {message}", flush=True)
        for sid in spot_ids:
            if 0 <= sid < NUM_SPOTS:
                agents[sid].post_user_message(message)

    def on_ask_user(message: str) -> None:
        print(f"\n[orch] {message}", flush=True)

    orch = OrchestratorLoop(
        client=orch_client,
        bus=orch_bus,
        on_route=on_route,
        on_ask_user=on_ask_user,
    )
    orch.start()

    # Stdin chat reader -> orchestrator (NOT directly to spots).
    def on_command(cmd: str) -> None:
        cmd = cmd.strip()
        if cmd in ("quit", "exit", "q"):
            print("[chat] quitting", flush=True)
            os._exit(0)
        if cmd.startswith("abort") or cmd.startswith("stop"):
            parts = cmd.split()
            targets: List[int] = []
            if len(parts) >= 2:
                try:
                    targets = [int(parts[1])]
                except ValueError:
                    targets = list(range(NUM_SPOTS))
            else:
                targets = list(range(NUM_SPOTS))
            for sid in targets:
                if 0 <= sid < NUM_SPOTS:
                    agents[sid].post_user_message(
                        "please stop whatever you are doing right now",
                    )
                    print(f"[chat] sent abort to spot{sid}", flush=True)
            return
        print(f"[chat] unknown command: {cmd!r}", flush=True)

    chat_reader = StdinChatReader(
        on_message=orch.post_user_message,
        on_command=on_command,
        prompt="\nyou> ",
    )
    chat_reader.start()

    print(
        "==> chat ready. type a goal (e.g. 'find a person'). "
        "use ':abort [N]' to stop, ':quit' to exit.",
        flush=True,
    )

    # ----- window + main loop -------------------------------------------
    window = MultiPaneWindow(
        pane_grid=[
            [POV_PANE_HW],
            [POV_PANE_HW],
            [COVERAGE_PANE_HW],
        ],
        title="mumt M-Agent.3 - chat agent (multi spot + orchestrator)",
    )

    primitive_ctxs = [
        ControllerCtx(
            sim=sim, spot_id=sid, teleop=teleops[sid], coverage=coverage,
            memory=memory,
            latest_camera_hfov_rad=math.radians(SPOT_HEAD_HFOV_DEG),
        )
        for sid in range(NUM_SPOTS)
    ]

    current: List[Optional[Controller]] = [None] * NUM_SPOTS
    current_name: List[Optional[str]] = [None] * NUM_SPOTS
    current_started_t: List[float] = [0.0] * NUM_SPOTS
    sim_t = 0.0

    pane_h, pane_w = COVERAGE_PANE_HW

    def render_coverage_pane() -> np.ndarray:
        fine = coverage.render_topdown(
            t_now=sim_t, spot_colors_bgr=SPOT_COVERAGE_BGR,
        )
        nz, nx = fine.shape[:2]
        scale = min(pane_h / nz, pane_w / nx)
        new_h = max(1, int(round(nz * scale)))
        new_w = max(1, int(round(nx * scale)))
        upscaled = cv2.resize(
            fine, (new_w, new_h), interpolation=cv2.INTER_NEAREST,
        )
        canvas = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
        y0 = (pane_h - new_h) // 2
        x0 = (pane_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = upscaled
        view = canvas[y0:y0 + new_h, x0:x0 + new_w]
        coverage.draw_coarse_grid(view, cell_pixel_scale=scale)
        coverage.draw_spot_markers(
            view,
            spot_poses=[
                (
                    float(spot_bodies[sid].translation.x),
                    float(spot_bodies[sid].translation.z),
                    teleops[sid].state.yaw,
                )
                for sid in range(NUM_SPOTS)
            ],
            spot_colors_bgr=SPOT_COVERAGE_BGR,
            cell_pixel_scale=scale,
        )
        return canvas

    try:
        # Prime with one frame.
        obs = sim.get_sensor_observations(head_agent_ids)
        rgb0 = obs[SPOT_0_HEAD_AGENT_ID][rgb_uuids[0]][:, :, :3]
        rgb1 = obs[SPOT_1_HEAD_AGENT_ID][rgb_uuids[1]][:, :, :3]
        window.show([
            (rgb0, ["spot0: idle", "(orchestrator awaiting input)"]),
            (rgb1, ["spot1: idle", "(orchestrator awaiting input)"]),
            (render_coverage_pane(), [], True),
        ])

        while not window.should_close():
            dt = window.tick(TARGET_FPS)
            state = window.poll_input()
            if state.quit_pressed:
                break
            sim_t += dt

            # ----- per-spot dispatcher / primitive plumbing ----------
            for sid in range(NUM_SPOTS):
                stop_reason = dispatcher.consume_stop(sid)
                if stop_reason is not None and current[sid] is not None:
                    current[sid].abort(stop_reason)

                if current[sid] is not None:
                    res = current[sid].step(dt, primitive_ctxs[sid])
                    if res is not None:
                        _print_primitive_result(sid, current_name[sid], res)
                        dispatcher.report_done(
                            sid, current_name[sid] or "?", res,
                        )
                        current[sid] = None
                        current_name[sid] = None
                else:
                    teleops[sid].drive(dt)

                if current[sid] is None and dispatcher.has_pending(sid):
                    req = dispatcher.try_start_pending(sid)
                    if req is not None:
                        new_ctl = req.controller
                        name = req.name
                        new_ctl.progress_cb = (
                            lambda payload, _name=name, _sid=sid:
                            dispatcher.push_progress(_sid, _name, payload)
                        )
                        new_ctl.start(primitive_ctxs[sid])
                        dispatcher.note_started(sid, name)
                        current[sid] = new_ctl
                        current_name[sid] = name
                        current_started_t[sid] = time.monotonic()
                        args_str = ", ".join(
                            f"{k}={v!r}" for k, v in (req.args or {}).items()
                        )
                        print(
                            f"\n[primitive] spot{sid} START "
                            f"{name}({args_str})",
                            flush=True,
                        )

            # ----- sensor obs + ctx refresh --------------------------
            obs = sim.get_sensor_observations(head_agent_ids)
            rgbs = []
            depths = []
            for sid in range(NUM_SPOTS):
                rgb = obs[head_agent_ids[sid]][rgb_uuids[sid]][:, :, :3]
                depth = obs[head_agent_ids[sid]][depth_uuids[sid]]
                rgbs.append(rgb)
                depths.append(depth)
                primitive_ctxs[sid].latest_rgb = rgb
                primitive_ctxs[sid].latest_rgb_is_bgr = False
                primitive_ctxs[sid].latest_depth = depth

            # ----- ambient captioner snapshots ----------------------
            sectors: List[Optional[str]] = []
            for sid in range(NUM_SPOTS):
                body_pos = spot_bodies[sid].translation
                sec = coverage.coarse_label_for_world_xz(
                    float(body_pos.x), float(body_pos.z),
                )
                sectors.append(sec)
                if sid < len(caption_workers):
                    caption_workers[sid].post_observation(
                        rgb=rgbs[sid], t_sim=sim_t, sector=sec,
                        pose_x=float(body_pos.x),
                        pose_z=float(body_pos.z),
                        pose_yaw_rad=float(teleops[sid].state.yaw),
                        rgb_is_bgr=False,
                    )

            # ----- coverage map update -------------------------------
            cells_stamped_total = 0
            for sid in range(NUM_SPOTS):
                body_pos = spot_bodies[sid].translation
                cam_T_world = CoverageMap.head_camera_world_transform(
                    sim, agent_id=head_agent_ids[sid],
                    sensor_uuid=depth_uuids[sid],
                )
                cells_stamped_total += coverage.update_from_depth(
                    spot_id=sid, t_now=sim_t,
                    cam_T_world=cam_T_world, depth=depths[sid],
                    hfov_deg=SPOT_HEAD_HFOV_DEG,
                    body_xz=(float(body_pos.x), float(body_pos.z)),
                )

            # ----- update agents' state snapshots --------------------
            cov_seen = int((coverage.last_seen_t.max(axis=-1) > -1e8).sum())
            cov_nav = int(coverage.is_navigable.sum())
            cov_pct = 100.0 * cov_seen / max(1, cov_nav)
            user_pos = human_ao.translation
            user_xz = (float(user_pos.x), float(user_pos.z))
            user_sector = coverage.coarse_label_for_world_xz(
                user_xz[0], user_xz[1],
            )
            with state_lock:
                for sid in range(NUM_SPOTS):
                    body_pos = spot_bodies[sid].translation
                    running_str: Optional[str] = None
                    if (
                        current[sid] is not None
                        and current_name[sid] is not None
                    ):
                        age = time.monotonic() - current_started_t[sid]
                        running_str = (
                            f"{current_name[sid]} ({age:.1f}s elapsed)"
                        )
                    snap = state_snapshots[sid]
                    snap["pose_xz"] = (
                        float(body_pos.x), float(body_pos.z),
                    )
                    snap["yaw_rad"] = float(teleops[sid].state.yaw)
                    snap["sector"] = sectors[sid]
                    snap["sim_t"] = float(sim_t)
                    snap["coverage_summary"] = (
                        f"shared coverage: {cov_seen}/{cov_nav} cells "
                        f"({cov_pct:.1f}%) seen"
                    )
                    snap["running_tool"] = running_str
                    snap["user_pose_xz"] = user_xz
                    snap["user_sector"] = user_sector

            # ----- HUD --------------------------------------------------
            hud_panes: list = []
            for sid in range(NUM_SPOTS):
                body_pos = spot_bodies[sid].translation
                running_str = None
                if (
                    current[sid] is not None
                    and current_name[sid] is not None
                ):
                    age = time.monotonic() - current_started_t[sid]
                    running_str = (
                        f"{current_name[sid]} ({age:.1f}s elapsed)"
                    )
                with state_lock:
                    speak = last_speak[sid].get("text") or ""
                    thinking = last_thinking[sid].get("text") or ""
                    action = last_action[sid].get("text") or ""
                    alert_text = last_alert[sid].get("text") or ""
                    alert_t = last_alert[sid].get("t")
                alert_age = (
                    time.monotonic() - alert_t
                    if alert_t is not None else 1e9
                )
                state_line = (
                    f"spot{sid}: running ({running_str})"
                    if current[sid] is not None
                    else f"spot{sid}: idle" if not agents[sid].is_running_task
                    else f"spot{sid}: thinking"
                )
                hud = [
                    f"spot{sid}  pose=({body_pos.x:+.2f}, {body_pos.z:+.2f}) "
                    f"yaw {math.degrees(teleops[sid].state.yaw):+5.1f}deg  "
                    f"sector={sectors[sid] or '--'}",
                    state_line,
                ]
                if alert_text and alert_age < 30.0:
                    hud.append(f"!! ALERT: {alert_text[:70]}")
                if speak:
                    hud.extend(_wrap(f"said: {speak}", 60))
                if thinking:
                    t_short = thinking.replace("\n", " ").strip()
                    if len(t_short) > 80:
                        t_short = t_short[:77] + "..."
                    hud.append(f"think: {t_short}")
                if action:
                    hud.append(f"act: {action[:60]}")
                hud_panes.append((rgbs[sid], hud))

            cov_pane = render_coverage_pane()
            cov_hud = [
                f"coverage map ({coverage.cfg.coarse_cell_m:g} m sectors) "
                f"{cov_seen}/{cov_nav} ({cov_pct:.0f}%) seen  "
                f"stamped {cells_stamped_total}  mem rows {len(memory)}",
                f"user@{user_sector or '--'}  "
                + "  ".join(
                    f"spot{sid}@{sectors[sid] or '--'}"
                    for sid in range(NUM_SPOTS)
                ),
                f"t={sim_t:6.1f}s  fps={1.0/max(1e-3, dt):4.1f}",
            ]
            window.show([*hud_panes, (cov_pane, cov_hud, True)])

    finally:
        for agent in agents:
            agent.stop(wait=False)
        orch.stop(wait=False)
        chat_reader.stop()
        for cw in caption_workers:
            cw.stop(timeout=2.0)
        if on_demand_captioner is not None:
            on_demand_captioner.stop(wait=False)
        if on_demand_detector is not None:
            on_demand_detector.stop(wait=False)
        if on_demand_recaller is not None:
            on_demand_recaller.stop(wait=False)
        memory.close()
        window.close()
        sim.close()


if __name__ == "__main__":
    main()
