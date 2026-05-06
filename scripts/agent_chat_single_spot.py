#!/usr/bin/env python
"""M-Agent.3 Slice A: single-Spot chat harness for the LLM autonomy stack.

Layout:
  +---------------------------------+
  |                                 |
  |     Spot 0 head POV (RGB)       |  top pane
  |     with HUD overlays:          |
  |       agent state               |
  |       last <speak>              |
  |       last <thinking>           |
  |       last action               |
  |                                 |
  +---------------------------------+
  |                                 |
  |     Coverage map (5 m sectors)  |  bottom pane
  |                                 |
  +---------------------------------+

User chats in the terminal:

    you> find a person in the kitchen
    spot> looking up where the kitchen is
    [agent spot0] step 1 (1.4s): recall(question='which sector is the kitchen?')
    spot> heading to C2 to look for someone
    [agent spot0] step 2 (1.1s): find(label='human', sector='C2')
    ...

Special chat commands:
    :abort        ask the agent to stop the running primitive
                  (delivered as a UserMessage; the agent decides)
    :quit         clean shutdown

The agent runs on its own thread and never touches habitat-sim. The
main thread owns the sim, ticks the controller installed in the
single primitive slot, and pushes pose/state snapshots into a
thread-safe dict the agent reads via a callback.
"""
from __future__ import annotations

import faulthandler
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

faulthandler.enable()

import cv2
import numpy as np

# cv2 / habitat_sim GLX init order: open + close a throwaway cv2 window
# BEFORE habitat_sim is imported, so cv2's GTK/GL state is bound first
# and habitat_sim's renderer doesn't poison it. Same dance as the
# two-spot teleop script.
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

# Sized in LOGICAL pixels for a 2x HiDPI Ubuntu desktop. The cv2
# canvas reports e.g. 480 px wide but the WM upscales 2x so it takes
# ~960 physical px on the right-screen column. Habitat still renders
# at 480x640 and we let cv2 downscale into the small panes so HiDPI
# text + sensor frames stay sharp on the display.
POV_PANE_HW = (270, 480)
COVERAGE_PANE_HW = (210, 480)

TARGET_FPS = 60
SPOT_HEAD_HFOV_DEG = 110.0
INITIAL_SPOT_PAN = 0.0
INITIAL_SPOT_TILT = math.radians(10.0)

# Single-spot tint for the coverage pane.
SPOT_COVERAGE_BGR = [(255, 255, 0)]  # cyan


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
    """Tiny word-wrapper for HUD overlays."""
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

    # Pick a spawn pair so we can put the human at the path midpoint;
    # we use only spot 0 but having a "target" pose for the human keeps
    # find/recall demos meaningful.
    print("==> picking spawn pair on navmesh", flush=True)
    p_spot_0, p_spot_other, geo_d = sample_far_pair_navmesh(
        sim.pathfinder, n_samples=600, min_clearance=0.7, refine_iters=3,
    )
    p_human = navmesh_path_midpoint(sim.pathfinder, p_spot_0, p_spot_other)
    print(
        f"    spot at {tuple(round(float(x), 3) for x in p_spot_0)}, "
        f"human at {tuple(round(float(x), 3) for x in p_human)}, "
        f"geodesic gap {geo_d:.2f}m",
        flush=True,
    )

    spot = add_kinematic_spot(
        sim, p_spot_0,
        yaw_rad=_yaw_facing(p_spot_0, (float(p_human[0]), float(p_human[2]))),
    )
    # Keep a handle to the humanoid so we can stream its pose into the
    # agent's state block as ``user:`` -- the chat user IS this human
    # in the scene (in the headless harness; in the VR rig the same
    # AO is the embodied operator).
    human_ao = add_kinematic_humanoid(
        sim, p_human,
        yaw_rad=_yaw_facing(p_human, (float(p_spot_0[0]), float(p_spot_0[2]))),
        name="female_0",
    )

    head = PanTiltHead(sim, agent_id=SPOT_0_HEAD_AGENT_ID, body_ao=spot)
    head.set_pan_tilt(pan=INITIAL_SPOT_PAN, tilt=INITIAL_SPOT_TILT)
    head.sync()

    teleop = SpotTeleop(sim, body_ao=spot, head=head, params=TeleopParams())
    rgb_uuid, depth_uuid = head_sensor_uuids(0)

    coverage = CoverageMap(
        sim=sim, n_spots=1,
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

    # ----- ambient captioner + on-demand pools ---------------------------
    memory_path = default_jsonl_path()
    memory = MemoryTable(jsonl_path=memory_path)
    print(f"==> memory table -> {memory_path}", flush=True)

    caption_worker: Optional[CaptionWorker] = None
    on_demand_captioner: Optional[OnDemandCaptioner] = None
    try:
        caption_client = GeminiClient(
            model=os.environ.get(
                "MUMT_CAPTION_MODEL", "gemini-3.1-flash-lite-preview",
            ),
        )
        caption_worker = CaptionWorker(
            spot_id=0, client=caption_client, memory=memory, period_s=2.0,
        )
        caption_worker.start()
        on_demand_captioner = OnDemandCaptioner(caption_client)
        print(
            f"==> captioner running (model={caption_client.model})",
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

    # ----- agent + dispatcher --------------------------------------------
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
    bus = EventBus(maxsize=128)

    # State snapshot accessible to the agent thread without touching the sim.
    state_lock = threading.Lock()
    state_snapshot: Dict[str, Any] = {
        "pose_xz": (0.0, 0.0),
        "yaw_rad": 0.0,
        "sector": None,
        "sim_t": 0.0,
        "coverage_summary": "",
        "running_tool": None,
        "user_pose_xz": None,
        "user_sector": None,
    }

    def get_state() -> Dict[str, Any]:
        with state_lock:
            return dict(state_snapshot)

    # User-facing callbacks: agent -> terminal.
    last_speak: Dict[str, Optional[str]] = {"text": None, "t": None}
    last_thinking: Dict[str, Optional[str]] = {"text": None, "t": None}
    last_action: Dict[str, Optional[str]] = {"text": None, "t": None}

    def on_speak(spot_id: int, text: str) -> None:
        # Print after a newline so the prompt redraw doesn't crowd the line.
        print(f"\nspot> {text}", flush=True)
        with state_lock:
            last_speak["text"] = text
            last_speak["t"] = time.monotonic()

    def on_thinking(spot_id: int, text: str) -> None:
        with state_lock:
            last_thinking["text"] = text
            last_thinking["t"] = time.monotonic()

    def on_action(spot_id: int, text: str) -> None:
        print(f"\nspot> [tool] {text}", flush=True)
        with state_lock:
            last_action["text"] = text
            last_action["t"] = time.monotonic()

    last_alert: Dict[str, Optional[Any]] = {"text": None, "t": None}

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
            last_alert["text"] = description
            last_alert["t"] = time.monotonic()

    agent = AgentLoop(
        spot_id=0,
        client=agent_client,
        dispatcher=dispatcher,
        bus=bus,
        coverage=coverage,
        get_state=get_state,
        on_demand_captioner=on_demand_captioner,
        on_demand_detector=on_demand_detector,
        on_demand_recaller=on_demand_recaller,
        on_speak=on_speak,
        on_thinking=on_thinking,
        on_action=on_action,
        on_alert=on_alert,
    )
    agent.start()

    # Stdin chat reader.
    def on_command(cmd: str) -> None:
        if cmd in ("quit", "exit", "q"):
            print("[chat] quitting", flush=True)
            os._exit(0)
        if cmd in ("abort", "stop"):
            agent.post_user_message(
                "please stop whatever you are doing right now",
            )
            return
        print(f"[chat] unknown command: {cmd!r}", flush=True)

    chat_reader = StdinChatReader(
        on_message=agent.post_user_message,
        on_command=on_command,
        prompt="\nyou> ",
    )
    chat_reader.start()

    print(
        "==> chat ready. type a goal (e.g. 'find a person'). "
        "use ':abort' to stop, ':quit' to exit.",
        flush=True,
    )

    # ----- window + main loop -------------------------------------------
    window = MultiPaneWindow(
        pane_grid=[[POV_PANE_HW], [COVERAGE_PANE_HW]],
        title="mumt M-Agent.3 - chat agent (single spot)",
    )

    primitive_ctx = ControllerCtx(
        sim=sim, spot_id=0, teleop=teleop, coverage=coverage,
        memory=memory,
        latest_camera_hfov_rad=math.radians(SPOT_HEAD_HFOV_DEG),
    )

    current: Optional[Controller] = None
    current_name: Optional[str] = None
    current_started_t: float = 0.0
    sim_t = 0.0

    pane_h, pane_w = COVERAGE_PANE_HW

    def render_coverage_pane() -> np.ndarray:
        """Render coverage at native resolution then aspect-preserving
        upscale into the pane; draw 5m grid + spot marker on top."""
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
        body_pos = spot.translation
        coverage.draw_spot_markers(
            view,
            spot_poses=[(float(body_pos.x), float(body_pos.z), teleop.state.yaw)],
            spot_colors_bgr=SPOT_COVERAGE_BGR,
            cell_pixel_scale=scale,
        )
        return canvas

    try:
        # Prime the window with one frame.
        sensor_ids = [SPOT_0_HEAD_AGENT_ID]
        obs = sim.get_sensor_observations(sensor_ids)
        rgb = obs[SPOT_0_HEAD_AGENT_ID][rgb_uuid][:, :, :3]
        window.show([
            (rgb, ["agent: idle", "(waiting for first goal)"]),
            (render_coverage_pane(), [], True),
        ])

        while not window.should_close():
            dt = window.tick(TARGET_FPS)
            state = window.poll_input()
            if state.quit_pressed:
                break
            sim_t += dt

            # ----- dispatcher: stop request --------------------------
            stop_reason = dispatcher.consume_stop(0)
            if stop_reason is not None and current is not None:
                current.abort(stop_reason)

            # ----- step current primitive ----------------------------
            if current is not None:
                res = current.step(dt, primitive_ctx)
                if res is not None:
                    _print_primitive_result(current_name, res)
                    dispatcher.report_done(0, current_name or "?", res)
                    current = None
                    current_name = None
            else:
                # Idle: hold position so the spot doesn't drift on residual cmd.
                teleop.drive(dt)

            # ----- install pending request ---------------------------
            if current is None and dispatcher.has_pending(0):
                req = dispatcher.try_start_pending(0)
                if req is not None:
                    new_ctl = req.controller
                    name = req.name
                    new_ctl.progress_cb = (
                        lambda payload, _name=name:
                        dispatcher.push_progress(0, _name, payload)
                    )
                    new_ctl.start(primitive_ctx)
                    dispatcher.note_started(0, name)
                    current = new_ctl
                    current_name = name
                    current_started_t = time.monotonic()
                    args_str = ", ".join(
                        f"{k}={v!r}" for k, v in (req.args or {}).items()
                    )
                    print(
                        f"\n[primitive] spot0 START {name}({args_str})",
                        flush=True,
                    )

            # ----- sensor obs + ctx refresh --------------------------
            obs = sim.get_sensor_observations(sensor_ids)
            rgb = obs[SPOT_0_HEAD_AGENT_ID][rgb_uuid][:, :, :3]
            depth = obs[SPOT_0_HEAD_AGENT_ID][depth_uuid]
            primitive_ctx.latest_rgb = rgb
            primitive_ctx.latest_rgb_is_bgr = False
            primitive_ctx.latest_depth = depth

            # ----- ambient captioner snapshot ------------------------
            body_pos = spot.translation
            sector = coverage.coarse_label_for_world_xz(
                float(body_pos.x), float(body_pos.z),
            )
            if caption_worker is not None:
                caption_worker.post_observation(
                    rgb=rgb, t_sim=sim_t, sector=sector,
                    pose_x=float(body_pos.x),
                    pose_z=float(body_pos.z),
                    pose_yaw_rad=float(teleop.state.yaw),
                    rgb_is_bgr=False,
                )

            # ----- coverage map update -------------------------------
            cam_T_world = CoverageMap.head_camera_world_transform(
                sim, agent_id=SPOT_0_HEAD_AGENT_ID, sensor_uuid=depth_uuid,
            )
            cells_stamped = coverage.update_from_depth(
                spot_id=0, t_now=sim_t,
                cam_T_world=cam_T_world, depth=depth,
                hfov_deg=SPOT_HEAD_HFOV_DEG,
                body_xz=(float(body_pos.x), float(body_pos.z)),
            )

            # ----- update agent's state snapshot ---------------------
            cov_seen = int((coverage.last_seen_t.max(axis=-1) > -1e8).sum())
            cov_nav = int(coverage.is_navigable.sum())
            cov_pct = 100.0 * cov_seen / max(1, cov_nav)
            running_str: Optional[str] = None
            if current is not None and current_name is not None:
                age = time.monotonic() - current_started_t
                running_str = (
                    f"{current_name} ({age:.1f}s elapsed)"
                )
            user_pos = human_ao.translation
            user_xz = (float(user_pos.x), float(user_pos.z))
            user_sector = coverage.coarse_label_for_world_xz(
                user_xz[0], user_xz[1],
            )
            with state_lock:
                state_snapshot["pose_xz"] = (
                    float(body_pos.x), float(body_pos.z),
                )
                state_snapshot["yaw_rad"] = float(teleop.state.yaw)
                state_snapshot["sector"] = sector
                state_snapshot["sim_t"] = float(sim_t)
                state_snapshot["coverage_summary"] = (
                    f"coverage: {cov_seen}/{cov_nav} cells "
                    f"({cov_pct:.1f}%) seen"
                )
                state_snapshot["running_tool"] = running_str
                state_snapshot["user_pose_xz"] = user_xz
                state_snapshot["user_sector"] = user_sector

            # ----- HUD --------------------------------------------------
            agent_state_line = (
                f"agent: running ({running_str})"
                if current is not None
                else "agent: idle" if not agent.is_running_task
                else "agent: thinking"
            )
            with state_lock:
                speak = last_speak.get("text") or ""
                thinking = last_thinking.get("text") or ""
                action = last_action.get("text") or ""
                alert_text = last_alert.get("text") or ""
                alert_t = last_alert.get("t")
            # Pin the alert in the HUD for 30 s after it fires.
            alert_age = (
                time.monotonic() - alert_t
                if alert_t is not None else 1e9
            )
            hud = [
                f"spot0  pose=({body_pos.x:+.2f}, {body_pos.z:+.2f}) "
                f"yaw {math.degrees(teleop.state.yaw):+5.1f}deg  "
                f"sector={sector or '--'}",
                agent_state_line,
            ]
            if alert_text and alert_age < 30.0:
                hud.append(f"!! ALERT: {alert_text[:80]}")
            if speak:
                hud.extend(_wrap(f"said: {speak}", 70))
            if thinking:
                # Truncate thinking aggressively; it's noisy.
                t_short = thinking.replace("\n", " ").strip()
                if len(t_short) > 90:
                    t_short = t_short[:87] + "..."
                hud.append(f"think: {t_short}")
            if action:
                hud.append(f"act: {action[:70]}")
            hud.append(
                f"t={sim_t:6.1f}s  fps={1.0/max(1e-3, dt):4.1f}  "
                f"cov {cov_seen}/{cov_nav} ({cov_pct:.0f}%)  "
                f"stamped {cells_stamped}  mem rows {len(memory)}"
            )

            cov_pane = render_coverage_pane()
            cov_hud = [
                f"coverage map ({coverage.cfg.coarse_cell_m:g} m sectors)",
                f"sector {sector or '--'}  "
                f"running {current_name or '-'}",
            ]
            window.show([
                (rgb, hud),
                (cov_pane, cov_hud, True),
            ])

    finally:
        agent.stop(wait=False)
        chat_reader.stop()
        if caption_worker is not None:
            caption_worker.stop(timeout=2.0)
        if on_demand_captioner is not None:
            on_demand_captioner.stop(wait=False)
        if on_demand_detector is not None:
            on_demand_detector.stop(wait=False)
        if on_demand_recaller is not None:
            on_demand_recaller.stop(wait=False)
        memory.close()
        window.close()
        sim.close()


def _print_primitive_result(name: Optional[str], result: PrimitiveResult) -> None:
    """Compact terminal trace; the agent gets the same payload via
    ``ToolResult`` events, but the human watching the terminal sees it
    here too in case the agent never narrates."""
    fx, fz, fyaw = result.final_pose
    name_disp = name or result.primitive
    print(
        f"\n[primitive] spot0 {name_disp} -> {result.status} "
        f"({result.reason}) t={result.t_elapsed_s:.1f}s "
        f"pose=({fx:+.2f},{fz:+.2f}, yaw {math.degrees(fyaw):+5.1f}deg)",
        flush=True,
    )


if __name__ == "__main__":
    main()
