#!/usr/bin/env python3
"""M3c custom Habitat-HITL app for the mumt project.

Spawns 2 kinematic Spots + 1 kinematic humanoid in HSSD scene 102344049 and
streams the world over the standard Habitat-HITL websocket protocol so the
``siro_hitl_unity_client`` Quest 2 APK can render it head-tracked.

Joystick handling (requires the matching Unity client extension that fills
the ``xr`` block of ClientState; see Milestone M3c phase C in the README):

- The user is embodied as the humanoid: per frame we snap the humanoid's
  XZ position to the headset pose reported by the client, and rotate the
  body to match the headset's horizontal facing. Camera comes from the
  headset itself (Unity-side), so the user is "inside" the avatar.
- The left thumbstick drives the user's locomotion. Each frame we
  integrate a velocity in the head's horizontal facing direction, clamp
  to the navmesh, then push the new target as a ``teleportAvatarBasePosition``
  keyframe message so the client shifts its XR origin accordingly. The
  current siro_hitl_unity_client (HEAD = dbfa5a6) honors that message
  via Assets/Scripts/AvatarPositionHandler.cs.
- The right thumbstick drives Spot 0: y-axis -> forward/back along the
  Spot's body forward, x-axis -> yaw rate. Movement is clamped against
  the navmesh via ``pathfinder.try_step``.

Run via ``scripts/07_run_mumt_hitl_server.sh``. The corresponding hydra
config lives in ``scripts/config/mumt_hitl.yaml``.
"""
from __future__ import annotations

import faulthandler
import math
import os
import sys
import threading
import time

# Capture C-level crashes (segfault / abort / fpe) and unblock-on-signal
# so we can pry open the next time the main process dies silently.
# - faulthandler.enable() prints a Python traceback when the process gets
#   SIGSEGV, SIGFPE, SIGABRT, etc. (instead of just exiting silently).
# - register(SIGUSR1) lets us trigger a live traceback dump from another
#   shell with ``kill -USR1 <pid>`` if the main loop merely hangs.
faulthandler.enable(file=sys.stderr, all_threads=True)
try:
    import signal as _sig  # noqa: PLC0415
    faulthandler.register(_sig.SIGUSR1, file=sys.stderr, all_threads=True)
except Exception:  # noqa: BLE001 -- Windows / restricted envs
    pass
from concurrent.futures import Future
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Sequence

import hydra
import magnum as mn
import numpy as np

import habitat_sim
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.key_mapping import KeyCode, XRButton
from habitat_hitl.core.user_mask import Mask

# Project root (one level above this script) must be on sys.path so the
# ``mumt_sim`` package imports work without a ``pip install -e .``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mumt_sim.agents import (  # noqa: E402
    _HUMAN_FORWARD_YAW_OFFSET,
    add_kinematic_humanoid,
    add_kinematic_spot,
)
from mumt_sim.spawn import (  # noqa: E402
    find_open_spawn_spot,
    sample_navmesh_cluster,
)
from mumt_sim.vr_displays import (  # noqa: E402
    AlertWedgeDisplay,
    DisplayLayout,
    DisplayManager,
    PointerKeyframe,
    SpotPovDisplay,
    TextDisplay,
    TopDownMapDisplay,
)
from mumt_sim.teleop import SpotTeleop, TeleopParams  # noqa: E402


# HUD modes the user cycles through with the right A button. Three
# rotating panels (status is permanent below). The autonomy integration
# (M3c phase E) reduced the cycle from 4 -> 3 because the user asked
# for "map / spot 0 feed / spot 1 feed" specifically.
_HUD_MODE_MAP: int = 0
_HUD_MODE_SPOT0_POV: int = 1
_HUD_MODE_SPOT1_POV: int = 2
_HUD_MODE_COUNT: int = 3


# Spot 0 teleop tuning (matches mumt_sim.teleop defaults). Mounted here so
# downstream tweaks don't drag a config-file dependency along.
_SPOT_FORWARD_SPEED: float = 0.8

# SMPL-X URDF places its root at the *pelvis*, not the feet. If we set the
# AO translation to the navmesh point's Y directly, the legs go below the
# floor (you only see the upper torso when looking down). Lifting the AO
# by ~1.0 m puts an adult-height pelvis at the right altitude so feet
# clear the floor. Habitat-lab's KinematicHumanoid uses base_offset=0.9
# but in our HSSD scenes the navmesh sits a hair above the visible floor,
# so we err on the side of a slightly higher lift; tune with
# ``mumt.humanoid_pelvis_lift_m`` if avatars look "stilt-walking".
_DEFAULT_HUMANOID_PELVIS_LIFT_M: float = 1.00

# Spot URDF root is at the body center; feet hang ~48 cm below. Same fix
# pattern as the humanoid: lift the AO so feet land on the navmesh.
# Habitat-lab's SpotRobot uses base_offset=(0, -0.48, 0); we mirror that.
_DEFAULT_SPOT_BASE_LIFT_M: float = 0.48
_SPOT_YAW_RATE: float = math.radians(90.0)
_HUMAN_WALK_SPEED: float = 1.0  # m/s; humans walk a touch faster than Spot
_THUMBSTICK_DEADZONE: float = 0.15

# Manual override release: how long the right stick has to sit at the
# deadzone before the LLM gets the spot back. Short enough to feel
# responsive when the user lets go, long enough to ride out brief stick
# returns to neutral while reorienting.
_OVERRIDE_RELEASE_S: float = 0.6


def _yaw_quat(yaw_rad: float) -> mn.Quaternion:
    return mn.Quaternion.rotation(mn.Rad(yaw_rad), mn.Vector3.y_axis())


def _yaw_facing(from_xyz: Sequence[float], to_xyz: Sequence[float]) -> float:
    """Yaw in our +Y/forward=+X convention pointing ``from_xyz`` at ``to_xyz``."""
    dx = float(to_xyz[0]) - float(from_xyz[0])
    dz = float(to_xyz[2]) - float(from_xyz[2])
    # +Y rotation: forward(yaw) = (cos yaw, 0, -sin yaw); we want forward
    # to align with (dx, ?, dz), so yaw = atan2(-dz, dx).
    return math.atan2(-dz, dx)


def _wrap_for_hud(text: str, width: int) -> List[str]:
    """Word-wrap ``text`` into chunks of <= ``width`` characters.

    Used by the per-Spot POV HUD overlay to fit "said: ..." lines
    inside the HUD strip without clipping. Newlines collapse to single
    spaces (the orchestrator's free-form text often contains them)."""
    text = (text or "").strip().replace("\n", " ")
    if not text:
        return []
    out: List[str] = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        out.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        out.append(text)
    return out


class _StubHeadState:
    """Pan/tilt placeholder. Autonomy primitives don't actively pan/tilt
    the head in the HITL build (we drive Spot kinematically through the
    body AO + body-mounted SpotHeadCam), so the angles stay at 0."""

    pan: float = 0.0
    tilt: float = 0.0


class _LiftedSpotTeleop(SpotTeleop):
    """SpotTeleop variant that applies a constant Y lift on push.

    The HITL app renders Spot AOs at ``navmesh_y + spot_base_lift_m``
    so feet clear the floor (URDF root is at the body, feet hang
    ~0.48 m below). The base ``SpotTeleop._push()`` writes the AO's
    translation directly from ``state.position``, whose Y has been
    snapped to the navmesh by ``pathfinder.try_step``. Without this
    override the kinematic Spot would sink into the floor every time
    the autonomy stack drove it.

    ``state.position`` itself stays at the navmesh-Y so the next
    ``try_step`` keeps working; only the AO write picks up the lift.
    """

    def __init__(self, *args, body_lift_y: float = 0.0, **kwargs) -> None:
        # ``SpotTeleop.__init__`` calls ``self._push()`` at the end of
        # construction, which on this subclass reads ``_body_lift_y`` --
        # so set it BEFORE delegating up.
        self._body_lift_y = float(body_lift_y)
        super().__init__(*args, **kwargs)
        # Re-anchor state.y to the navmesh level so the lift maths is
        # consistent. ``__init__`` captured the lifted Y from the AO;
        # subtract the lift now.
        self.state.position = mn.Vector3(
            self.state.position.x,
            self.state.position.y - self._body_lift_y,
            self.state.position.z,
        )

    def _push(self) -> None:  # type: ignore[override]
        self.body_ao.translation = mn.Vector3(
            self.state.position.x,
            self.state.position.y + self._body_lift_y,
            self.state.position.z,
        )
        self.body_ao.rotation = mn.Quaternion.rotation(
            mn.Rad(self.state.yaw), mn.Vector3.y_axis(),
        )
        self.head.set_pan_tilt(pan=self.state.pan, tilt=self.state.tilt)
        self.head.sync()


class _StubPanTiltHead:
    """Minimal duck-typed PanTiltHead replacement.

    ``mumt_sim.teleop.SpotTeleop`` expects a ``PanTiltHead``-shaped
    object with ``state.pan/tilt``, ``set_pan_tilt(pan, tilt)`` and
    ``sync()``. The non-HITL scripts use a real PanTiltHead bound to a
    habitat-sim Agent (the head sensors live on a separate Agent in
    ``mumt_sim.scene.make_sim``). The HITL SimDriver doesn't expose
    that layout, and our ``SpotHeadCam`` parks sensors directly on the
    spot AO's root scene node, so head pose follows the body
    automatically -- pan/tilt is a no-op here.
    """

    def __init__(self) -> None:
        self.state = _StubHeadState()

    def set_pan_tilt(self, pan: float, tilt: float) -> None:
        self.state.pan = float(pan)
        self.state.tilt = float(tilt)

    def sync(self) -> None:
        return  # body-mounted sensors track the AO automatically


class AppStateMumt(AppState):
    """Custom HITL app: 2 Spots + 1 VR-embodied humanoid in HSSD."""

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._gui_input = app_service.gui_input

        # Activate the single remote user (Quest headset).
        app_service.users.activate_user(0)

        cfg = self._app_service.config.mumt
        self._dataset = cfg.dataset
        self._scene = cfg.scene
        # Pelvis / base lifts in metres; tune at the CLI if avatars look
        # stilted or sunken (e.g. mumt.humanoid_pelvis_lift_m=0.95).
        self._humanoid_pelvis_lift_m: float = float(
            getattr(cfg, "humanoid_pelvis_lift_m",
                    _DEFAULT_HUMANOID_PELVIS_LIFT_M)
        )
        self._spot_base_lift_m: float = float(
            getattr(cfg, "spot_base_lift_m", _DEFAULT_SPOT_BASE_LIFT_M)
        )
        self._app_service.reconfigure_sim(self._dataset, self._scene)

        sim = self._app_service.sim
        # HSSD has no cached navmesh; recompute including static furniture.
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        # Spawn the human at an open floor-level point in the navmesh.
        # ``find_open_spawn_spot`` picks one at random from the top 10
        # candidates by clearance, so each restart gives a fresh location
        # -- useful because some HSSD couches are floor-level and slip
        # past ``distance_to_closest_obstacle`` (the seat geometry is at
        # floor Y with a wide free hemisphere above it). If a given
        # spawn lands the user "in" a couch, just restart the server.
        human_xyz, human_clearance = find_open_spawn_spot(
            sim.pathfinder, min_clearance=1.0, n_samples=2000, top_k=10,
        )
        self._humanoid_spawn: List[float] = list(human_xyz)

        # Cluster Spot spawns around the human's open spot. The cluster
        # sampler also asks for n=3 because it always seeds the first point
        # at a random navmesh sample; we discard that and re-anchor the
        # remaining two points on the human's hand-picked open spot.
        cluster_pts = sample_navmesh_cluster(
            sim.pathfinder, n=3, min_sep=1.2, cluster_radius=3.0,
        )
        # cluster_pts[0] is a random anchor; cluster_pts[1:] are 1.2-3 m
        # from it. Translating them so they're 1.2-3 m from our open spot
        # is "good enough" for the demo (yes the absolute distances drift
        # slightly, but the navmesh-snap below fixes them up).
        anchor = np.asarray(cluster_pts[0])
        offset = np.asarray(human_xyz) - anchor
        self._spot0_spawn: List[float] = list(
            sim.pathfinder.snap_point(np.asarray(cluster_pts[1]) + offset)
        )
        self._spot1_spawn: List[float] = list(
            sim.pathfinder.snap_point(np.asarray(cluster_pts[2]) + offset)
        )

        spawn_pts = [
            self._humanoid_spawn, self._spot0_spawn, self._spot1_spawn,
        ]
        centroid = np.mean(np.asarray(spawn_pts), axis=0).tolist()
        humanoid_yaw = _yaw_facing(self._humanoid_spawn, centroid)
        spot0_yaw = _yaw_facing(self._spot0_spawn, centroid)
        spot1_yaw = _yaw_facing(self._spot1_spawn, centroid)
        print(
            f"[mumt-hitl] human spawn clearance = {human_clearance:.2f} m",
            flush=True,
        )

        # Spawn kinematic articulated objects (no physics, fixed_base).
        # Lift each AO root so feet land on the navmesh (URDF roots are at
        # the body for Spot / pelvis for the humanoid, both above the feet).
        spot0_spawn_lifted = [
            self._spot0_spawn[0],
            self._spot0_spawn[1] + self._spot_base_lift_m,
            self._spot0_spawn[2],
        ]
        spot1_spawn_lifted = [
            self._spot1_spawn[0],
            self._spot1_spawn[1] + self._spot_base_lift_m,
            self._spot1_spawn[2],
        ]
        self._humanoid_ao = add_kinematic_humanoid(
            sim, self._humanoid_spawn, yaw_rad=humanoid_yaw,
        )
        self._spot0_ao = add_kinematic_spot(
            sim, spot0_spawn_lifted, yaw_rad=spot0_yaw,
        )
        self._spot1_ao = add_kinematic_spot(
            sim, spot1_spawn_lifted, yaw_rad=spot1_yaw,
        )

        # Mutable Spot 0 teleop state, driven by right thumbstick.
        # Stored at the *lifted* (rendered) Y so per-frame writes match
        # the spawn position; ``try_step`` is fed the same frame so its
        # navmesh snap stays consistent.
        self._spot0_pos: mn.Vector3 = mn.Vector3(*spot0_spawn_lifted)
        self._spot0_yaw_rad: float = spot0_yaw

        # Mutable user locomotion state. Driven by left thumbstick, fed back
        # to the client via ``teleportAvatarBasePosition`` keyframe messages.
        # Y is sticky to the humanoid spawn floor height; the Unity handler
        # ignores Y delta anyway.
        self._user_target_pos: mn.Vector3 = mn.Vector3(*self._humanoid_spawn)
        self._last_left_print_sec: int = 0
        # Set in on_environment_reset and cleared once we've actually told
        # the client to move; avoids stranding the user inside furniture at
        # whatever spot their Quest's tracking origin happened to be.
        self._needs_initial_teleport: bool = True
        self._teleport_attempts: int = 0
        # Initial humanoid yaw is what we fall back to if no head pose has
        # been received yet (cold connect).
        self._humanoid_yaw_rad: float = humanoid_yaw

        # Filled per-frame in sim_update.
        self._cam_transform: mn.Matrix4 = mn.Matrix4.identity_init()

        print(
            f"[mumt-hitl] spawned humanoid at {self._humanoid_spawn}, "
            f"spot0 at {self._spot0_spawn}, spot1 at {self._spot1_spawn}",
            flush=True,
        )

        # ---------------- Autonomy stack -----------------------------
        # Per-Spot RGB+depth head sensors. These also provide the color
        # raster the HUD POV displays render from, so we don't pay for
        # redundant per-spot sensors.
        self._head_cams: List[Any] = []
        self._teleops: List[SpotTeleop] = []
        self._head_stubs: List[_StubPanTiltHead] = []
        self._coverage = None
        # The coverage→map overlay wiring needs both ``_coverage``
        # (created in ``_setup_autonomy``) and ``_map_display`` (created
        # in the displays block below). They're constructed in
        # different orders, so we initialise the handle here and wire
        # them together once both exist.
        self._map_display = None
        self._memory = None
        self._caption_workers: List[Any] = []
        self._on_demand_captioner = None
        self._on_demand_detector = None
        self._on_demand_recaller = None
        self._agents: List[Any] = []
        self._spot_buses: List[Any] = []
        self._dispatcher = None
        self._orchestrator = None
        self._orchestrator_bus = None
        self._primitive_ctxs: List[Any] = []
        self._current_ctlrs: List[Optional[Any]] = []
        self._current_names: List[Optional[str]] = []
        self._current_started_t: List[float] = []
        self._state_lock = threading.Lock()
        self._state_snapshots: List[Dict[str, Any]] = []
        self._sim_t: float = 0.0
        self._autonomy_enabled: bool = False
        self._cv2_available: bool = False
        self._stt = None
        self._ptt_recorder = None
        self._ptt_active: bool = False
        self._ptt_pending: List[Future] = []
        self._ptt_point_cache: List[Any] = []
        # Manual override: any right-stick movement while a Spot POV is
        # in the main HUD slot takes over that Spot from the LLM. Idle
        # for ``_OVERRIDE_RELEASE_S`` seconds and the LLM gets it back.
        # ``_last_override_stick_t`` tracks the most recent non-zero
        # right-stick sample so the release logic can measure idle time.
        self._manual_override_active: bool = False
        self._manual_override_sid: Optional[int] = None
        self._last_override_stick_t: float = monotonic()
        # Latest pointer raycast hit -- updated while the right index
        # trigger is held; cleared when the trigger is released. PTT
        # release (right B up) consumes whatever's here at that moment.
        self._latest_pointer_hit = None
        self._last_speak: List[Dict[str, Optional[Any]]] = []
        self._last_thinking: List[Dict[str, Optional[Any]]] = []
        self._last_action: List[Dict[str, Optional[Any]]] = []
        self._last_alert: List[Dict[str, Optional[Any]]] = []
        # Wall-clock deadline (monotonic seconds) until which the alert
        # wedge for each Spot stays visible. 0.0 = inactive. Auto-hide
        # happens in sim_update each tick.
        self._alert_until_t: List[float] = [0.0, 0.0]
        self._alert_visible_dur_s: float = 5.0

        try:
            import cv2  # noqa: F401
            self._cv2_available = True
        except Exception:  # noqa: BLE001
            self._cv2_available = False

        autonomy_cfg = getattr(cfg, "autonomy", None)
        autonomy_enabled = bool(
            autonomy_cfg is not None
            and getattr(autonomy_cfg, "enabled", False)
        )
        self._autonomy_enabled = autonomy_enabled

        if autonomy_enabled:
            self._setup_autonomy(sim, autonomy_cfg)

        # ---------------- Virtual displays --------------------------
        # DisplayManager owns the wire protocol; we just register
        # providers. Layouts use anchor="head" -> HUD style, quad
        # parented to the XR camera so panels follow the user's gaze.
        # Offsets are in Unity-local space (X=right, Y=up, Z=forward).
        self._displays = DisplayManager(self._app_service.client_message_manager)

        # Laser pointer line: separate keyframe block (mumtPointer) that
        # the Unity client renders as a LineRenderer + endpoint sphere.
        # Driven by ``_handle_pointing`` while the right index trigger is
        # held; flushed once per frame in ``sim_update``.
        self._pointer_kf = PointerKeyframe(
            self._app_service.client_message_manager,
        )
        # Color used while pointing. Red sticks out against arbitrary
        # scene materials and matches the agent-message "(user pointed
        # to ...)" affordance the orchestrator already understands.
        self._pointer_color: Tuple[float, float, float] = (0.95, 0.20, 0.20)
        # Fallback ray length when the controller doesn't hit anything
        # (e.g. user points at the sky). Keeps the line visible so the
        # user has continuous visual feedback that pointing is active.
        self._pointer_fallback_dist_m: float = 25.0

        # Re-broadcast "create" events whenever a new client connects.
        # Without this, clients that connect *after* the first sim_update
        # never see the create message (it lives in the per-frame outgoing
        # message dict, which is cleared each tick after dispatch).
        remote = self._app_service.remote_client_state
        if remote is not None:
            remote.on_client_connected.registerCallback(
                lambda _record: self._displays.on_scene_change()
            )

        # Main HUD slot: large quad in the user's front view. The 3-way
        # cycle (map -> spot 0 POV -> spot 1 POV) toggles which provider
        # is visible here; the others stay hidden but keep rendering
        # off-screen so a flip is instant.
        self._main_hud_layout = DisplayLayout(
            anchor="head",
            offset=(0.0, -0.10, 0.85),
            rot_euler_deg=(0.0, 0.0, 0.0),
            size=(0.55, 0.42),
        )
        # Permanent always-on status strip just above the main panel.
        # Thin so it doesn't fight the main HUD for screen real estate.
        self._status_hud_layout = DisplayLayout(
            anchor="head",
            offset=(0.0, +0.20, 0.85),
            rot_euler_deg=(0.0, 0.0, 0.0),
            size=(0.55, 0.13),
        )
        self._displays.add(
            TextDisplay(
                uid="status",
                layout=self._status_hud_layout,
                text_fn=self._status_text,
                size_hw=(192, 768),
                font_px=22,
                fps=4.0,
            ),
        )
        try:
            map_disp = TopDownMapDisplay(
                uid="map",
                layout=self._main_hud_layout,
                sim=sim,
                pose_fn=self._agent_poses_for_map,
                size_hw=(512, 512),
                cell_m=0.10,
                fps=5.0,
            )
            self._displays.add(map_disp)
            self._map_display = map_disp
        except RuntimeError as exc:
            print(f"[mumt-hitl] TopDownMapDisplay disabled: {exc}", flush=True)

        # Wire the live CoverageMap (set up earlier in ``_setup_autonomy``
        # when autonomy is enabled) into the top-down HUD so it draws
        # sectors + per-spot coverage tints + body markers, matching
        # ``scripts/agent_chat_multi_spot.py``'s coverage pane. Spot
        # colours match the marker colours used in
        # ``_agent_poses_for_map`` (yellow Spot0, magenta Spot1).
        if self._map_display is not None and self._coverage is not None:
            try:
                n_spots = int(self._coverage.n_spots)
                self._map_display.set_coverage_overlay(
                    coverage_fn=lambda: self._coverage,
                    sim_t_fn=lambda: float(self._sim_t),
                    spot_colors_bgr=[
                        (0, 220, 255),    # Spot0 -- yellow in BGR
                        (220, 80, 255),   # Spot1 -- magenta in BGR
                    ][:n_spots],
                )
                print(
                    "[mumt-hitl] map: CoverageMap overlay wired "
                    f"(n_spots={n_spots})", flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[mumt-hitl] map: coverage overlay wiring failed "
                    f"({exc!r})", flush=True,
                )

        # Spot 0 / Spot 1 first-person POVs. We borrow the COLOR sensor
        # from SpotHeadCam if autonomy is on; otherwise SpotPovDisplay
        # creates its own sensor (the previous behaviour pre-autonomy).
        # Each POV gets a center crosshair plus a HUD text overlay
        # showing pose / sector / running tool / last-said line --
        # mirrors the per-Spot HUD in scripts/agent_chat_multi_spot.py.
        for sid, ao in enumerate([self._spot0_ao, self._spot1_ao]):
            existing = None
            if (
                autonomy_enabled
                and sid < len(self._head_cams)
                and self._head_cams[sid] is not None
            ):
                existing = self._head_cams[sid].color_sensor
            try:
                self._displays.add(
                    SpotPovDisplay(
                        uid=f"spot{sid}_pov",
                        layout=self._main_hud_layout,
                        sim=sim,
                        spot_ao=ao,
                        size_hw=(384, 512),
                        hfov_deg=90.0,
                        fps=15.0,
                        existing_sensor=existing,
                        draw_crosshair=True,
                        hud_text_fn=self._make_pov_text_fn(sid),
                        hud_font_px=18,
                    ),
                )
            except (ValueError, RuntimeError) as exc:
                print(
                    f"[mumt-hitl] SpotPovDisplay spot{sid} disabled: {exc}",
                    flush=True,
                )

        # Alert wedges. Spot 0 -> left edge of FOV, Spot 1 -> right
        # edge. Both start hidden; the on_alert callback registered in
        # _setup_autonomy flips visibility on (and sim_update flips it
        # back off after _alert_visible_dur_s). The wedge itself blinks
        # at ~3 Hz so it grabs peripheral attention even when held on.
        for sid, side_x in ((0, -0.55), (1, +0.55)):
            wedge_uid = f"alert_spot{sid}"
            try:
                self._displays.add(
                    AlertWedgeDisplay(
                        uid=wedge_uid,
                        layout=DisplayLayout(
                            anchor="head",
                            offset=(side_x, 0.0, 0.85),
                            rot_euler_deg=(0.0, 0.0, 0.0),
                            size=(0.06, 0.55),
                        ),
                        size_hw=(256, 32),
                        blink_hz=3.0,
                        fps=10.0,
                    ),
                )
                self._displays.set_visible(wedge_uid, False)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[mumt-hitl] alert wedge spot{sid} disabled: {exc}",
                    flush=True,
                )

        self._hud_mode: int = _HUD_MODE_MAP
        self._apply_hud_mode()

    def _agent_poses_for_map(self):
        """Return ``(world_x, world_z, yaw, color_rgb)`` for each visible
        agent so TopDownMapDisplay can draw markers."""
        h = self._humanoid_ao.translation
        s0 = self._spot0_ao.translation
        s1 = self._spot1_ao.translation
        return [
            # Cyan = user, yellow = Spot0 (driveable), magenta = Spot1.
            (float(h.x), float(h.z), self._humanoid_yaw_rad, (0, 220, 255)),
            (float(s0.x), float(s0.z), self._spot0_yaw_rad, (255, 220, 0)),
            (float(s1.x), float(s1.z), 0.0, (255, 80, 220)),
        ]

    _HUD_MODE_NAMES = {
        _HUD_MODE_MAP: "map",
        _HUD_MODE_SPOT0_POV: "spot0_pov",
        _HUD_MODE_SPOT1_POV: "spot1_pov",
    }

    def _apply_hud_mode(self) -> None:
        """Push current HUD mode through to display visibility.

        The status panel is always-on (fixed strip above the main HUD);
        only the main slot rotates between map / spot0 POV / spot1 POV.
        """
        is_map = self._hud_mode == _HUD_MODE_MAP
        is_s0 = self._hud_mode == _HUD_MODE_SPOT0_POV
        is_s1 = self._hud_mode == _HUD_MODE_SPOT1_POV
        if self._displays.has("map"):
            self._displays.set_visible("map", is_map)
        if self._displays.has("spot0_pov"):
            self._displays.set_visible("spot0_pov", is_s0)
        if self._displays.has("spot1_pov"):
            self._displays.set_visible("spot1_pov", is_s1)

    def _cycle_hud_mode(self) -> None:
        self._hud_mode = (self._hud_mode + 1) % _HUD_MODE_COUNT
        self._apply_hud_mode()
        print(
            f"[mumt-hitl] HUD mode -> {self._HUD_MODE_NAMES[self._hud_mode]}",
            flush=True,
        )

    def _status_text(self) -> str:
        # Lazy bind here so we don't capture stale references in __init__
        # (sim state mutates every frame). Cheap to recompute at 4 Hz.
        h_pos = self._humanoid_ao.translation
        s0_pos = self._spot0_ao.translation
        s1_pos = self._spot1_ao.translation
        lines: List[str] = []
        mode_name = self._HUD_MODE_NAMES.get(self._hud_mode, "?")
        active_sid = self._active_spot_id()
        if self._manual_override_active and active_sid is not None:
            ovr = f" [MANUAL S{active_sid}]"
        else:
            ovr = ""
        ptt = " [REC]" if self._ptt_active else ""
        ptr = " [POINT]" if self._latest_pointer_hit is not None else ""
        lines.append(f"MUMT VR  HUD={mode_name}{ovr}{ptt}{ptr}")
        lines.append(
            f"U({h_pos.x:+.1f},{h_pos.z:+.1f}) "
            f"S0({s0_pos.x:+.1f},{s0_pos.z:+.1f}) "
            f"S1({s1_pos.x:+.1f},{s1_pos.z:+.1f})"
        )
        if self._autonomy_enabled and len(self._last_speak) >= 2:
            for sid in (0, 1):
                speak = (self._last_speak[sid] or {}).get("text") or ""
                if speak:
                    if len(speak) > 60:
                        speak = speak[:57] + "..."
                    lines.append(f"S{sid}> {speak}")
        if self._autonomy_enabled:
            lines.append("Rstk-click cyc HUD  A=ctrl B=PTT trig=point")
        else:
            lines.append("L-stick walk  R-stick drive S0  Rstk-click HUD")
        return "\n".join(lines)

    def get_sim(self):
        return self._app_service.sim

    def on_environment_reset(self, episode_recorder_dict):
        # Re-emit display "create" events on next tick because the client
        # tears down its scene state when it sees a sceneChanged signal.
        if hasattr(self, "_displays"):
            self._displays.on_scene_change()
        client_msg = self._app_service.client_message_manager
        if client_msg is not None:
            # Notify the client that the scene mesh changed.
            client_msg.signal_scene_change(destination_mask=Mask.ALL)
            # Schedule a one-shot teleport of the user to the humanoid spawn
            # so they don't appear inside random furniture wherever the
            # Quest's tracking origin happens to be. We can't call
            # rebase_xr_headset_position() because the deployed
            # siro_hitl_unity_client (HEAD = dbfa5a6) only honors the
            # legacy teleportAvatarBasePosition key. Inject directly.
            #
            # Why not call this here unconditionally? AvatarPositionHandler
            # diff's the target against the *current* camera position, which
            # is identity until the first head pose arrives. We delay until
            # we've heard from the client; see _maybe_send_initial_teleport.
            self._needs_initial_teleport = True
            self._teleport_attempts = 0

    def _embody_humanoid_from_head_pose(self) -> None:
        """Snap the humanoid AO to wherever the user's headset is reporting.

        The user's Unity-side camera comes from the Quest's tracker, so by
        co-locating the humanoid's body with the headset we get free
        embodiment: the user is "inside" the avatar without needing
        first-person camera plumbing on the server.
        """
        remote = self._app_service.remote_client_state
        if remote is None:
            return  # headless server still spinning up / no networking

        _, head_rot = remote.get_head_pose(0)
        if head_rot is None:
            return  # no client connected yet; leave humanoid at spawn pose

        # IMPORTANT: We deliberately do NOT use the headset's reported world
        # position to place the humanoid. The headset position is in the
        # user's physical-room (Quest origin) frame, which we shift to the
        # right *world* position via ``teleportAvatarBasePosition`` keyframe
        # messages. Those messages take a frame to propagate client-side,
        # so reading head_pos this frame yields the pre-teleport position.
        # If we used it here, we'd drag the humanoid back to wherever the
        # user's Quest origin happens to be (often inside furniture) on
        # the first frame after every teleport.
        #
        # Instead we use ``self._user_target_pos`` -- the server's
        # authoritative model of where the user *should* be in world
        # space. It's set by ``_maybe_send_initial_teleport`` to the
        # humanoid spawn, and integrated forward by
        # ``_drive_user_from_left_thumbstick``. Headset-reported pose only
        # contributes *yaw*, so the user can still turn their head freely.

        # Recompute yaw from head's forward direction in world XZ. The head
        # quaternion returned by RemoteClientState is already in habitat
        # agent-space (z+ forward), so we rotate the (0, 0, 1) vector instead
        # of (0, 0, -1) as we'd do for raw Unity poses.
        forward = head_rot.transform_vector(mn.Vector3(0.0, 0.0, 1.0))
        forward.y = 0.0
        if forward.length() > 1e-3:
            forward = forward.normalized()
            self._humanoid_yaw_rad = math.atan2(-forward.z, forward.x)

        self._humanoid_ao.translation = mn.Vector3(
            float(self._user_target_pos.x),
            float(self._user_target_pos.y) + self._humanoid_pelvis_lift_m,
            float(self._user_target_pos.z),
        )
        # VR embodiment fix: when the user is "inside" the humanoid we want
        # the avatar's face (rendered via the SMPL-X mesh) to point in the
        # same world direction as the user's gaze. Empirically the URDF
        # head bone faces opposite of what the agents._HUMAN_FORWARD_YAW_OFFSET
        # comment claims, so without this extra +pi the user ends up looking
        # OUT THE BACK of the avatar's head. This stays local to the VR
        # embodiment path so non-VR rendering scripts keep their existing
        # +X = yaw=0 convention.
        self._humanoid_ao.rotation = _yaw_quat(
            self._humanoid_yaw_rad + _HUMAN_FORWARD_YAW_OFFSET + math.pi
        )

    def _maybe_send_initial_teleport(self) -> None:
        """Snap the user to ``_humanoid_spawn`` once the client connects.

        AvatarPositionHandler.cs diffs the target against the current camera
        position to compute the delta to apply to the XR origin. We resend
        the teleport every frame until either (a) we've confirmed the
        client's reported head_pos is close to the spawn (delta < 0.5 m),
        or (b) we've spammed for more than 60 frames (~1 s) without
        convergence -- at which point we assume the client is happy with
        whatever it ended up at and stop.

        Resending is necessary because the first teleport message often
        races with the client's first head_pose reports: the client may
        receive teleport(A) but report head_pos = B (its pre-teleport
        Quest-origin position) for one or two frames before the XR origin
        actually shifts. Without the loop the server would think the user
        is still at B and drag the humanoid (or, with the fix above, just
        stay confused about where the user is).
        """
        if not self._needs_initial_teleport:
            return
        remote = self._app_service.remote_client_state
        if remote is None:
            return  # headless server still spinning up / no networking
        head_pos, _ = remote.get_head_pose(0)
        if head_pos is None:
            return  # client not reporting yet; wait
        client_msg = self._app_service.client_message_manager
        if client_msg is None:
            return

        spawn_v = mn.Vector3(*self._humanoid_spawn)
        dx = float(head_pos.x) - spawn_v.x
        dz = float(head_pos.z) - spawn_v.z
        delta_xz = math.sqrt(dx * dx + dz * dz)

        # Send teleport (idempotent on the client; delta of 0 = no-op).
        for user_index in self._app_service.users.indices(Mask.ALL):
            msg = client_msg.get_messages()[user_index]
            msg["teleportAvatarBasePosition"] = [
                float(self._humanoid_spawn[0]),
                float(self._humanoid_spawn[1]),
                float(self._humanoid_spawn[2]),
            ]
        self._user_target_pos = spawn_v

        self._teleport_attempts += 1
        if delta_xz < 0.5:
            print(
                f"[mumt-hitl] user at spawn (|d|={delta_xz:.2f} m after "
                f"{self._teleport_attempts} attempts)",
                flush=True,
            )
            self._needs_initial_teleport = False
        elif self._teleport_attempts >= 60:
            print(
                f"[mumt-hitl] giving up on initial teleport "
                f"(|d|={delta_xz:.2f} m after {self._teleport_attempts} "
                f"attempts); user remains where the client last reported",
                flush=True,
            )
            self._needs_initial_teleport = False
        elif self._teleport_attempts == 1:
            print(
                f"[mumt-hitl] teleporting user to spawn {self._humanoid_spawn}"
                f" (|d|={delta_xz:.2f} m)",
                flush=True,
            )

    def _drive_user_from_left_thumbstick(self, dt: float) -> None:
        """Translate the user's XR origin by the left-thumbstick velocity.

        Frame sequence:
          1. Read the left-stick (x, y) from the rich XR input.
          2. Compute a horizontal velocity in the headset's facing frame
             (so 'stick up' = 'go where I'm looking').
          3. Integrate, clamp to the navmesh.
          4. Emit ``teleportAvatarBasePosition`` so the client shifts its
             XR origin to match. The Unity-side handler diff's this against
             the current camera position and applies the delta to the
             ``xrOriginNode`` GameObject.

        No-op until both an XR thumbstick and a head pose have been
        received from the client.
        """
        if dt <= 0.0:
            return

        sim = self._app_service.sim
        remote = self._app_service.remote_client_state
        if remote is None:
            return
        head_pos, head_rot = remote.get_head_pose(0)
        if head_pos is None:
            return

        xr = remote.get_xr_input(0)
        lx, ly = xr.left_controller.get_thumbstick()

        # Debug: print non-zero thumbstick values once per second to confirm
        # the rich xr block is reaching the server. Remove once teleop works.
        now = int(monotonic())
        if (lx != 0.0 or ly != 0.0) and now != self._last_left_print_sec:
            rx_, ry_ = xr.right_controller.get_thumbstick()
            print(
                f"[mumt-hitl] left=({lx:+.2f},{ly:+.2f}) "
                f"right=({rx_:+.2f},{ry_:+.2f})",
                flush=True,
            )
            self._last_left_print_sec = now

        if abs(lx) < _THUMBSTICK_DEADZONE:
            lx = 0.0
        if abs(ly) < _THUMBSTICK_DEADZONE:
            ly = 0.0

        if lx == 0.0 and ly == 0.0:
            # Stick is idle. Do *not* snap _user_target_pos to head_pos:
            # head_pos is in the user's physical-room (Quest origin) frame
            # and may not yet match the world position we just teleported
            # them to. Leaving _user_target_pos as-is keeps the server
            # authoritative; physical room-scale walking won't move the
            # avatar (acceptable trade-off to avoid the couch-snap bug).
            return

        forward = head_rot.transform_vector(mn.Vector3(0.0, 0.0, 1.0))
        forward.y = 0.0
        if forward.length() < 1e-3:
            return  # head is staring straight up/down; skip this frame
        forward = forward.normalized()
        # Right-hand rule: right = up x forward (we want stick.x > 0 to
        # strafe to the user's right when they look down +Z agent-space).
        right = mn.math.cross(mn.Vector3(0.0, 1.0, 0.0), forward).normalized()

        # Stick y forward, stick x strafe right.
        # Empirically (M3c phase D testing) the forward axis was reversed --
        # pushing the stick forward drove the user backward -- while strafe
        # was correct. Negating ly here keeps strafe behaviour and makes
        # "push forward = walk toward what you're looking at".
        velocity = forward * (_HUMAN_WALK_SPEED * -ly) + right * (
            _HUMAN_WALK_SPEED * lx
        )

        attempted = self._user_target_pos + velocity * dt
        clamped = mn.Vector3(
            sim.pathfinder.try_step(self._user_target_pos, attempted)
        )
        self._user_target_pos = clamped

        # Keyframe message: tell the client to teleport the avatar's base
        # to this world position. Y is ignored client-side; we still pass
        # the navmesh-snapped Y so the field is well-formed.
        client_msg = self._app_service.client_message_manager
        if client_msg is not None:
            for user_index in self._app_service.users.indices(Mask.ALL):
                msg = client_msg.get_messages()[user_index]
                msg["teleportAvatarBasePosition"] = [
                    float(clamped.x),
                    float(clamped.y),
                    float(clamped.z),
                ]

    def _drive_active_spot_from_right_thumbstick(self, dt: float) -> None:
        """Apply right-thumbstick (x = yaw, y = forward) to whichever
        Spot is currently active.

        Two implementation paths share this entrypoint:

        - Autonomy off: legacy direct AO mutation on Spot 0 (kept for
          the pure-VR demo path that doesn't pull in the agent stack;
          the right stick always drives Spot 0 in that mode).
        - Autonomy on + manual override held + a Spot is in the POV
          slot: route through that Spot's ``SpotTeleop.drive`` so the
          autonomy-side teleop state stays in lockstep with the AO.
          Otherwise this is a no-op (the LLM is driving).
        """
        if dt <= 0.0:
            return

        sim = self._app_service.sim
        remote = self._app_service.remote_client_state
        if remote is None:
            return
        xr = remote.get_xr_input(0)
        rx, ry = xr.right_controller.get_thumbstick()

        if abs(rx) < _THUMBSTICK_DEADZONE:
            rx = 0.0
        if abs(ry) < _THUMBSTICK_DEADZONE:
            ry = 0.0
        stick_active = (rx != 0.0) or (ry != 0.0)

        # ----------------------------------------------------------
        # Auto manual-override:
        #   * right-stick non-zero + HUD on a Spot POV
        #     -> engage override on the displayed spot, stop its LLM.
        #   * stick idle for ``_OVERRIDE_RELEASE_S`` seconds
        #     -> release back to the LLM.
        # No A-button required (UX feedback: any joystick movement is
        # an explicit user intent to take over).
        # ----------------------------------------------------------
        if self._autonomy_enabled:
            active_sid = self._active_spot_id()
            now = monotonic()
            if stick_active and active_sid is not None:
                self._last_override_stick_t = now
                if (
                    not self._manual_override_active
                    or self._manual_override_sid != active_sid
                ):
                    self._manual_override_active = True
                    self._manual_override_sid = active_sid
                    if self._dispatcher is not None:
                        try:
                            self._dispatcher.request_stop(
                                active_sid, reason="user manual override",
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[mumt-hitl] override request_stop "
                                f"warn: {exc!r}", flush=True,
                            )
                    print(
                        f"[mumt-hitl] manual override -> Spot{active_sid}",
                        flush=True,
                    )
            elif self._manual_override_active and not stick_active:
                idle_s = now - getattr(self, "_last_override_stick_t", now)
                if idle_s >= _OVERRIDE_RELEASE_S:
                    self._manual_override_active = False
                    self._manual_override_sid = None
                    print(
                        "[mumt-hitl] manual override: released "
                        f"(idle {idle_s:.1f}s)", flush=True,
                    )

        if self._autonomy_enabled and not self._manual_override_active:
            return
        if self._autonomy_enabled:
            sid = self._active_spot_id()
            if sid is None:
                return

        if self._autonomy_enabled and self._teleops:
            sid = self._active_spot_id()
            if sid is None:
                return
            forward_mps = _SPOT_FORWARD_SPEED * ry
            # rx > 0 (push right) -> turn right; teleop convention is
            # "yaw_left positive", same as our +Y rotation, so negate rx.
            yaw_rps = -_SPOT_YAW_RATE * rx
            try:
                self._teleops[sid].drive(
                    dt,
                    forward_mps=forward_mps,
                    lateral_mps=0.0,
                    yaw_rps=yaw_rps,
                )
            except Exception as exc:  # noqa: BLE001
                # Don't let a bad ao mutation crash the whole sim;
                # log throttled (once per second).
                last = getattr(self, "_last_drive_err_sec", -1)
                cur_sec = int(self._sim_t)
                if cur_sec - last >= 1:
                    print(
                        f"[mumt-hitl] override drive warn (Spot{sid}): "
                        f"{exc!r}", flush=True,
                    )
                    self._last_drive_err_sec = cur_sec
                return
            if sid == 0:
                self._spot0_pos = mn.Vector3(self._spot0_ao.translation)
                self._spot0_yaw_rad = float(self._teleops[0].state.yaw)
            return

        if ry != 0.0:
            speed = _SPOT_FORWARD_SPEED * ry
            cy = math.cos(self._spot0_yaw_rad)
            sy = math.sin(self._spot0_yaw_rad)
            forward_world = mn.Vector3(cy, 0.0, -sy)
            # ``try_step`` snaps its output Y to the navmesh and ignores any
            # lift we'd applied to ``_spot0_pos``; pull the navmesh-level
            # Y back down before stepping, then re-apply the lift to the
            # snapped result.
            unlifted_start = mn.Vector3(
                self._spot0_pos.x,
                self._spot0_pos.y - self._spot_base_lift_m,
                self._spot0_pos.z,
            )
            attempted = unlifted_start + forward_world * (speed * dt)
            stepped = sim.pathfinder.try_step(unlifted_start, attempted)
            self._spot0_pos = mn.Vector3(
                float(stepped[0]),
                float(stepped[1]) + self._spot_base_lift_m,
                float(stepped[2]),
            )

        if rx != 0.0:
            # rx > 0 (push right) -> turn right (yaw decreases in our +Y conv).
            self._spot0_yaw_rad -= _SPOT_YAW_RATE * rx * dt

        self._spot0_ao.translation = self._spot0_pos
        self._spot0_ao.rotation = _yaw_quat(self._spot0_yaw_rad)

    def _poll_buttons(self) -> None:
        """Per-frame discrete-button handling.

        Quest right-controller mapping (see InputTrackerXR.cs):
          - thumbstick click (``XRButton.PRIMARY_THUMBSTICK``):
                cycle the main HUD slot through map / spot 0 POV /
                spot 1 POV.
          - A (``XRButton.ONE``):
                while held, the user takes manual control of the spot
                currently shown in the POV slot (right thumbstick
                drives it; the LLM is told to stop). On the map, the
                button is a no-op.
          - B (``XRButton.TWO``):
                while held, push-to-talk: open the host mic, accumulate
                audio, and on release transcribe via ElevenLabs and
                post to the orchestrator. If the index trigger was held
                during PTT, the latest pointer hit is appended.
          - index trigger (``XRButton.PRIMARY_INDEX_TRIGGER``):
                pointing only. While held, the server raycasts from the
                right controller into the scene (and onto the HUD map
                quad if the map is showing) and stores the latest hit.

        Quest left-controller mapping is unchanged:
          - X (``XRButton.ONE`` on left controller):
                re-trigger the teleport-to-humanoid loop. Useful when
                the initial teleport races / drops on connect.
        """
        remote = self._app_service.remote_client_state
        if remote is None:
            return
        xr = remote.get_xr_input(0)

        # Once-per-second diagnostic so we can confirm whether the rich XR
        # block is actually reaching us. Prints stick axes, controller in-hand
        # flags, and head-pose availability. Helps diagnose "buttons don't
        # work" against APK / handshake regressions.
        now_sec = int(monotonic())
        if now_sec != getattr(self, "_last_input_diag_sec", -1):
            self._last_input_diag_sec = now_sec
            try:
                hp, _ = remote.get_head_pose(0)
            except Exception:  # noqa: BLE001
                hp = None
            lx, ly = xr.left_controller.get_thumbstick()
            rx, ry = xr.right_controller.get_thumbstick()
            l_in = xr.left_controller.get_is_controller_in_hand()
            r_in = xr.right_controller.get_is_controller_in_hand()
            it_l = xr.left_controller.get_index_trigger()
            it_r = xr.right_controller.get_index_trigger()
            try:
                cs = remote.get_recent_client_state_by_history_index(0, 0)
                cs_keys = sorted(cs.keys()) if cs else []
            except Exception:  # noqa: BLE001
                cs_keys = ["<err>"]
            print(
                f"[mumt-hitl/in] head={'OK' if hp is not None else 'NONE'} "
                f"L(stick={lx:+.2f},{ly:+.2f} hand={l_in} trig={it_l:.2f}) "
                f"R(stick={rx:+.2f},{ry:+.2f} hand={r_in} trig={it_r:.2f}) "
                f"keys={cs_keys}",
                flush=True,
            )

        # Edge-event diag: log every button down/up on either controller
        # so we can spot button-routing regressions (e.g. PTT not firing
        # because the server isn't seeing B-button events).
        for hand_label, ctrl in (
            ("L", xr.left_controller), ("R", xr.right_controller),
        ):
            for btn in list(XRButton):
                try:
                    if ctrl.get_button_down(btn):
                        print(
                            f"[mumt-hitl/btn] {hand_label}.{btn.name} DOWN",
                            flush=True,
                        )
                    if ctrl.get_button_up(btn):
                        print(
                            f"[mumt-hitl/btn] {hand_label}.{btn.name} UP",
                            flush=True,
                        )
                except Exception:  # noqa: BLE001
                    pass

        if xr.right_controller.get_button_down(XRButton.PRIMARY_THUMBSTICK):
            self._cycle_hud_mode()
            # Map mode has no active spot -> if user lets go of A here,
            # manual override silently no-ops; if they're still holding
            # A from a POV mode, we want to release on the spot they
            # were just controlling.
            if (
                self._manual_override_active
                and self._active_spot_id() is None
            ):
                self._manual_override_active = False
                print(
                    "[mumt-hitl] manual override released "
                    "(switched to map view)", flush=True,
                )

        if xr.left_controller.get_button_down(XRButton.ONE):
            self._needs_initial_teleport = True
            self._teleport_attempts = 0
            print("[mumt-hitl] manual teleport-to-humanoid requested",
                  flush=True)

        if self._autonomy_enabled:
            # Manual override is now driven by right-stick activity
            # (see ``_drive_active_spot_from_right_thumbstick``); the
            # A button is reserved for future use.

            self._handle_pointing(xr)
            self._handle_ptt(xr)

    def sim_update(self, dt: float, post_sim_update_dict):
        if self._gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        self._sim_t += float(dt)

        # Order matters: locomotion mutates client message state for THIS
        # frame; embodiment reads the latest head pose (which lags the
        # locomotion by one round-trip but converges within one tick).
        self._maybe_send_initial_teleport()
        self._poll_buttons()
        self._drive_user_from_left_thumbstick(dt)
        self._embody_humanoid_from_head_pose()
        self._drive_active_spot_from_right_thumbstick(dt)

        if self._autonomy_enabled:
            # Per-Spot autonomy lifecycle: stop -> step current
            # controller -> install pending. The active spot (the one
            # currently shown in POV) is skipped while the user holds
            # manual override, so the LLM doesn't fight the joystick.
            override_sid = (
                self._active_spot_id()
                if self._manual_override_active else None
            )
            for sid in range(len(self._teleops)):
                if sid == override_sid:
                    continue
                self._step_autonomy_per_spot(sid, dt)

            # Refresh sensors + ctxs + coverage + state snapshots once
            # all motion is settled this tick.
            self._refresh_autonomy_observations()
            self._refresh_state_snapshots()
            self._poll_pending_transcriptions()
            self._expire_alert_wedges()

        # Render any due virtual-display frames AFTER the agents have been
        # repositioned, so the Spot POV camera (attached to spot0_ao's
        # scene node) renders from the post-step pose.
        self._displays.tick()
        # Pointer line: edge-triggered. set_visible(...) was already
        # called above by ``_handle_pointing``; this just flushes the
        # current state into the per-user keyframe message.
        self._pointer_kf.tick()

        # In VR mode the Unity client uses its own headset camera, so the
        # server-side cam_transform is cosmetic (it would only affect the
        # server's own GUI window, which we run headless). Identity is fine.
        post_sim_update_dict["cam_transform"] = self._cam_transform

    # ------------------------------------------------------------------
    # Autonomy stack setup and per-frame helpers
    # ------------------------------------------------------------------

    def _setup_autonomy(self, sim, autonomy_cfg) -> None:
        """Bring up the head sensors + agent stack.

        Best-effort: every optional service (caption / detect / recall /
        STT) is wrapped in its own try/except so a missing API key or
        unavailable Jetson doesn't break the basic VR + manual demo.
        """
        # Lazy imports keep this whole tree optional: when autonomy is
        # disabled, none of the heavy deps need to be installed.
        from mumt_sim.agent.coverage import (  # noqa: PLC0415
            CoverageMap, CoverageMapConfig,
        )
        from mumt_sim.agent.detection import (  # noqa: PLC0415
            OnDemandDetector, YoloeClient,
        )
        from mumt_sim.agent.head_cam import SpotHeadCam  # noqa: PLC0415
        from mumt_sim.agent.loop import (  # noqa: PLC0415
            AgentClient, AgentLoop, EventBus, ToolDispatcher,
        )
        from mumt_sim.agent.memory import (  # noqa: PLC0415
            MemoryTable, default_jsonl_path,
        )
        from mumt_sim.agent.orchestrator import (  # noqa: PLC0415
            OrchestratorClient, OrchestratorLoop,
        )
        from mumt_sim.agent.perception import (  # noqa: PLC0415
            CaptionWorker, GeminiClient, OnDemandCaptioner,
        )
        from mumt_sim.agent.recall import (  # noqa: PLC0415
            OnDemandRecaller, RecallClient,
        )
        from mumt_sim.agent.tools import ControllerCtx  # noqa: PLC0415

        n_spots = 2
        self._head_stubs = [_StubPanTiltHead() for _ in range(n_spots)]

        # Per-Spot RGB+depth sensors mounted on the AO root scene node.
        # Same offset/orientation/HFOV as ``mumt_sim.scene`` defaults.
        for sid, ao in enumerate([self._spot0_ao, self._spot1_ao]):
            self._head_cams.append(SpotHeadCam(
                sim, ao, spot_id=sid,
            ))

        # Teleops: one per Spot. The body AO is the lifted (rendered)
        # AO; the stub head means pan/tilt is a no-op. Pass the
        # ``spot_base_lift_m`` so the teleop adds it back when writing
        # the AO every frame -- without that the ``pathfinder.try_step``
        # navmesh-Y would sink the rendered Spot into the floor.
        for sid, ao in enumerate([self._spot0_ao, self._spot1_ao]):
            tp = _LiftedSpotTeleop(
                sim, body_ao=ao, head=self._head_stubs[sid],
                params=TeleopParams(),
                body_lift_y=self._spot_base_lift_m,
            )
            self._teleops.append(tp)

        # Coverage map. The overlay wiring into the HUD top-down map
        # happens later in ``__init__`` after the display has been
        # constructed (the displays block runs *after* ``_setup_autonomy``
        # finishes, so we can't grab ``_map_display`` here yet).
        self._coverage = CoverageMap(
            sim=sim, n_spots=n_spots,
            config=CoverageMapConfig(
                fine_cell_m=0.10, max_range_m=5.0, pixel_stride=4,
            ),
        )

        # Memory + ambient captioner. Wrapped because GEMINI_API_KEY
        # may be missing.
        self._memory = MemoryTable(jsonl_path=default_jsonl_path())
        try:
            caption_model = os.environ.get(
                "MUMT_CAPTION_MODEL",
                str(getattr(autonomy_cfg, "caption_model",
                            "gemini-3.1-flash-lite-preview")),
            )
            caption_client = GeminiClient(model=caption_model)
            for sid in range(n_spots):
                cw = CaptionWorker(
                    spot_id=sid, client=caption_client,
                    memory=self._memory, period_s=2.0,
                )
                cw.start()
                self._caption_workers.append(cw)
            self._on_demand_captioner = OnDemandCaptioner(
                caption_client, max_workers=6,
            )
            print(
                f"[mumt-hitl] captioner up (model={caption_client.model})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[mumt-hitl] captioning disabled ({exc}). "
                f"Set GEMINI_API_KEY to enable.",
                flush=True,
            )

        # YOLOE find detector.
        try:
            yc = YoloeClient(
                base_url=os.environ.get(
                    "MUMT_YOLOE_URL",
                    str(getattr(autonomy_cfg, "yoloe_url", None) or "")
                    or None,
                ),
                timeout_s=4.0,
            )
            self._on_demand_detector = OnDemandDetector(yc, max_workers=4)
            print(
                f"[mumt-hitl] YOLOE detector up ({yc.base_url})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[mumt-hitl] YOLOE detector disabled ({exc})", flush=True)

        # Recall LLM.
        try:
            recall_model = os.environ.get(
                "MUMT_RECALL_MODEL",
                str(getattr(autonomy_cfg, "recall_model",
                            "gemini-3.1-flash-lite-preview")),
            )
            rc = RecallClient(model=recall_model)
            self._on_demand_recaller = OnDemandRecaller(rc, max_workers=2)
            print(
                f"[mumt-hitl] recall LLM up (model={rc.model})", flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[mumt-hitl] recall disabled ({exc}). "
                f"Set GEMINI_API_KEY to enable.",
                flush=True,
            )

        # Per-spot AgentLoops + dispatcher.
        try:
            agent_model = os.environ.get(
                "MUMT_AGENT_MODEL",
                str(getattr(autonomy_cfg, "agent_model",
                            "gemini-3.1-flash-lite-preview")),
            )
            agent_client = AgentClient(model=agent_model)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[mumt-hitl] AgentClient init failed ({exc}); autonomy "
                f"loops disabled. Set GEMINI_API_KEY to enable.",
                flush=True,
            )
            return

        self._dispatcher = ToolDispatcher()
        self._state_snapshots = [
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
            for _ in range(n_spots)
        ]
        self._last_speak = [
            {"text": None, "t": None} for _ in range(n_spots)
        ]
        self._last_thinking = [
            {"text": None, "t": None} for _ in range(n_spots)
        ]
        self._last_action = [
            {"text": None, "t": None} for _ in range(n_spots)
        ]
        self._last_alert = [
            {"text": None, "t": None} for _ in range(n_spots)
        ]

        for sid in range(n_spots):
            bus = EventBus(maxsize=128)

            def _make_get_state(_sid: int):
                def _get_state() -> Dict[str, Any]:
                    with self._state_lock:
                        return dict(self._state_snapshots[_sid])
                return _get_state

            def _on_speak(_sid: int, text: str) -> None:
                print(f"\n[autonomy] spot{_sid}> {text}", flush=True)
                with self._state_lock:
                    self._last_speak[_sid]["text"] = text
                    self._last_speak[_sid]["t"] = time.monotonic()

            def _on_thinking(_sid: int, text: str) -> None:
                with self._state_lock:
                    self._last_thinking[_sid]["text"] = text
                    self._last_thinking[_sid]["t"] = time.monotonic()

            def _on_action(_sid: int, text: str) -> None:
                print(f"\n[autonomy] spot{_sid}> [tool] {text}", flush=True)
                with self._state_lock:
                    self._last_action[_sid]["text"] = text
                    self._last_action[_sid]["t"] = time.monotonic()

            def _on_alert(_sid: int, description: str) -> None:
                print(
                    f"\n[autonomy] !!! ALERT spot{_sid}: {description}",
                    flush=True,
                )
                with self._state_lock:
                    self._last_alert[_sid]["text"] = description
                    self._last_alert[_sid]["t"] = time.monotonic()
                # Trigger the on-FOV-edge wedge. Auto-hide deadline is
                # checked every tick in sim_update.
                self._alert_until_t[_sid] = (
                    time.monotonic() + self._alert_visible_dur_s
                )
                wedge_uid = f"alert_spot{_sid}"
                if self._displays.has(wedge_uid):
                    self._displays.set_visible(wedge_uid, True)

            agent = AgentLoop(
                spot_id=sid,
                client=agent_client,
                dispatcher=self._dispatcher,
                bus=bus,
                coverage=self._coverage,
                get_state=_make_get_state(sid),
                on_demand_captioner=self._on_demand_captioner,
                on_demand_detector=self._on_demand_detector,
                on_demand_recaller=self._on_demand_recaller,
                on_speak=_on_speak,
                on_thinking=_on_thinking,
                on_action=_on_action,
                on_alert=_on_alert,
            )
            agent.start()
            self._spot_buses.append(bus)
            self._agents.append(agent)

        self._primitive_ctxs = [
            ControllerCtx(
                sim=sim,
                spot_id=sid,
                teleop=self._teleops[sid],
                coverage=self._coverage,
                memory=self._memory,
                latest_camera_hfov_rad=self._head_cams[sid].hfov_rad,
            )
            for sid in range(n_spots)
        ]
        self._current_ctlrs = [None] * n_spots
        self._current_names = [None] * n_spots
        self._current_started_t = [0.0] * n_spots

        # Orchestrator (single LLM that fans user messages out).
        try:
            orch_model = os.environ.get(
                "MUMT_ORCH_MODEL",
                str(getattr(autonomy_cfg, "orchestrator_model",
                            "gemini-3.1-flash-lite-preview")),
            )
            orch_client = OrchestratorClient(
                num_spots=n_spots, model=orch_model,
            )

            def _on_route(spot_ids: Sequence[int], message: str) -> None:
                ids_str = ",".join(str(i) for i in spot_ids)
                print(f"\n[orch] -> spot[{ids_str}]: {message}", flush=True)
                for sid in spot_ids:
                    if 0 <= sid < n_spots:
                        self._agents[sid].post_user_message(message)

            def _on_ask_user(message: str) -> None:
                print(f"\n[orch] {message}", flush=True)

            self._orchestrator_bus = EventBus(maxsize=64)
            self._orchestrator = OrchestratorLoop(
                client=orch_client,
                bus=self._orchestrator_bus,
                on_route=_on_route,
                on_ask_user=_on_ask_user,
            )
            self._orchestrator.start()
            print(
                f"[mumt-hitl] orchestrator up "
                f"(model={orch_client.model})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[mumt-hitl] orchestrator disabled ({exc}). "
                f"Set GEMINI_API_KEY to enable.",
                flush=True,
            )

        # Speech-to-text + push-to-talk recorder. Both optional.
        stt_cfg = getattr(autonomy_cfg, "stt", None)
        if stt_cfg is not None and bool(getattr(stt_cfg, "enabled", False)):
            try:
                from mumt_sim.agent.voice import (  # noqa: PLC0415
                    ElevenLabsSTT, GeminiSTT, PushToTalkRecorder,
                )
                # Backend selection. Defaults to ``gemini`` so the same
                # GEMINI_API_KEY that powers the agents is reused for
                # transcription -- avoids a second paid subscription
                # and survives ElevenLabs free-tier blocks. Set
                # ``mumt.autonomy.stt.backend: elevenlabs`` to keep the
                # old Scribe path.
                stt_backend = str(
                    getattr(stt_cfg, "backend", "gemini")
                ).lower()
                if stt_backend == "elevenlabs":
                    self._stt = ElevenLabsSTT(
                        model=str(getattr(stt_cfg, "model", "scribe_v1")),
                    )
                elif stt_backend == "gemini":
                    self._stt = GeminiSTT(
                        model=str(
                            getattr(
                                stt_cfg, "model",
                                "gemini-3.1-flash-lite-preview",
                            )
                        ),
                    )
                else:
                    raise RuntimeError(
                        f"unknown stt.backend {stt_backend!r}; "
                        f"expected 'gemini' or 'elevenlabs'"
                    )
                # Optional explicit ALSA/PulseAudio device override. Some
                # default-device setups (e.g. PulseAudio "default" with
                # 32 advertised channels) crash PortAudio on the first
                # InputStream open from a habitat-sim host. Pinning the
                # device avoids that. ``stt.device`` accepts an int index
                # or a substring of the device's name.
                device_cfg = getattr(stt_cfg, "device", None)
                self._ptt_recorder = PushToTalkRecorder(
                    sample_rate=int(getattr(stt_cfg, "sample_rate", 16000)),
                    device=device_cfg,
                )
                # Pre-warm: open + immediately close the stream once at
                # startup so PortAudio negotiates with the host audio
                # daemon now, while nothing else is touching the sim.
                # If this fails, we disable PTT cleanly here instead of
                # killing the sim mid-frame on the first B press.
                try:
                    self._ptt_recorder.start()
                    self._ptt_recorder.stop()
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[mumt-hitl] PTT pre-warm failed: {exc!r}. "
                        f"Disabling push-to-talk for this session. "
                        f"Try setting mumt.autonomy.stt.device to an "
                        f"explicit input device index or name.",
                        flush=True,
                    )
                    self._ptt_recorder = None
                stt_label = (
                    f"{type(self._stt).__name__}/{self._stt.model}"
                )
                if self._ptt_recorder is not None:
                    print(
                        f"[mumt-hitl] STT ({stt_label}) "
                        f"+ PTT recorder up "
                        f"(device={device_cfg!r})", flush=True,
                    )
                else:
                    print(
                        f"[mumt-hitl] STT ({stt_label}) "
                        f"up; PTT disabled", flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[mumt-hitl] STT disabled ({exc}). "
                    f"For Gemini STT set GEMINI_API_KEY and install "
                    f"'google-genai'; for ElevenLabs set "
                    f"ELEVENLABS_API_KEY and install 'elevenlabs'.",
                    flush=True,
                )
                self._stt = None
                self._ptt_recorder = None

    def _make_pov_text_fn(self, sid: int) -> Callable[[], List[str]]:
        """Build a HUD-text callable for the spot{sid} POV display.

        Returns a closure that the SpotPovDisplay calls on every render
        tick. The closure reads the per-Spot state snapshot + last
        speak/thinking/action/alert dicts under the same lock the
        AgentLoop callbacks write under, so no torn reads.
        """

        def _fn() -> List[str]:
            with self._state_lock:
                snap = (
                    dict(self._state_snapshots[sid])
                    if sid < len(self._state_snapshots)
                    else {}
                )
                speak = (
                    self._last_speak[sid]
                    if sid < len(self._last_speak) else {}
                )
                thinking = (
                    self._last_thinking[sid]
                    if sid < len(self._last_thinking) else {}
                )
                action = (
                    self._last_action[sid]
                    if sid < len(self._last_action) else {}
                )
                alert = (
                    self._last_alert[sid]
                    if sid < len(self._last_alert) else {}
                )

            lines: List[str] = []
            pose_xz = snap.get("pose_xz") or (0.0, 0.0)
            yaw_deg = math.degrees(float(snap.get("yaw_rad") or 0.0))
            sector = snap.get("sector") or "--"
            running = snap.get("running_tool")
            head1 = (
                f"S{sid} "
                f"({pose_xz[0]:+.1f},{pose_xz[1]:+.1f}) "
                f"yaw{yaw_deg:+.0f}d "
                f"sec={sector}"
            )
            lines.append(head1)

            if running:
                lines.append(f"run: {running}")
            else:
                # Distinguish "thinking" (LLM busy) from idle so the user
                # knows the agent is alive even when no tool is running.
                running_task = False
                if sid < len(self._agents):
                    try:
                        running_task = bool(
                            self._agents[sid].is_running_task
                        )
                    except Exception:  # noqa: BLE001
                        running_task = False
                lines.append("status: thinking" if running_task else "status: idle")

            now = time.monotonic()
            alert_t = alert.get("t")
            alert_text = alert.get("text") or ""
            if (
                alert_text
                and alert_t is not None
                and (now - float(alert_t)) < 30.0
            ):
                lines.append(f"!! ALERT: {alert_text[:48]}")

            speak_text = speak.get("text") or ""
            if speak_text:
                lines.extend(_wrap_for_hud(f"said: {speak_text}", 38))

            thinking_text = (thinking.get("text") or "").replace("\n", " ")
            if thinking_text:
                if len(thinking_text) > 48:
                    thinking_text = thinking_text[:45] + "..."
                lines.append(f"think: {thinking_text}")

            action_text = action.get("text") or ""
            if action_text:
                if len(action_text) > 40:
                    action_text = action_text[:37] + "..."
                lines.append(f"act: {action_text}")
            return lines

        return _fn

    def _expire_alert_wedges(self) -> None:
        """Hide each alert wedge once its visibility window elapses."""
        if not self._alert_until_t:
            return
        now = time.monotonic()
        for sid, until in enumerate(self._alert_until_t):
            if until > 0.0 and now >= until:
                self._alert_until_t[sid] = 0.0
                wedge_uid = f"alert_spot{sid}"
                if self._displays.has(wedge_uid):
                    self._displays.set_visible(wedge_uid, False)

    # --- per-frame helpers ---------------------------------------------

    def _step_autonomy_per_spot(self, sid: int, dt: float) -> None:
        """Advance one Spot's controller-step + dispatcher cycle."""
        if self._dispatcher is None:
            return
        stop_reason = self._dispatcher.consume_stop(sid)
        if stop_reason is not None and self._current_ctlrs[sid] is not None:
            self._current_ctlrs[sid].abort(stop_reason)

        if self._current_ctlrs[sid] is not None:
            res = self._current_ctlrs[sid].step(
                dt, self._primitive_ctxs[sid],
            )
            if res is not None:
                self._dispatcher.report_done(
                    sid, self._current_names[sid] or "?", res,
                )
                self._current_ctlrs[sid] = None
                self._current_names[sid] = None
        else:
            # Idle drive so the body holds position cleanly (zero
            # velocities, but pushes state into the AO each tick).
            self._teleops[sid].drive(dt)

        if (
            self._current_ctlrs[sid] is None
            and self._dispatcher.has_pending(sid)
        ):
            req = self._dispatcher.try_start_pending(sid)
            if req is not None:
                new_ctl = req.controller
                name = req.name
                new_ctl.progress_cb = (
                    lambda payload, _name=name, _sid=sid:
                    self._dispatcher.push_progress(_sid, _name, payload)
                )
                new_ctl.start(self._primitive_ctxs[sid])
                self._dispatcher.note_started(sid, name)
                self._current_ctlrs[sid] = new_ctl
                self._current_names[sid] = name
                self._current_started_t[sid] = time.monotonic()

    def _refresh_autonomy_observations(self) -> None:
        """Render head sensors, fill ControllerCtx, update coverage,
        post ambient caption snapshots."""
        if self._coverage is None or not self._head_cams:
            return

        for sid, head_cam in enumerate(self._head_cams):
            rgb = head_cam.render_rgb()
            depth = head_cam.render_depth()
            if rgb is None or depth is None:
                continue
            ctx = self._primitive_ctxs[sid]
            ctx.latest_rgb = rgb
            ctx.latest_rgb_is_bgr = False
            ctx.latest_depth = depth
            ctx.latest_camera_hfov_rad = head_cam.hfov_rad

            body_pos = self._teleops[sid].state.position
            body_xz = (float(body_pos.x), float(body_pos.z))
            try:
                self._coverage.update_from_depth(
                    spot_id=sid,
                    t_now=self._sim_t,
                    cam_T_world=head_cam.cam_T_world(),
                    depth=depth,
                    hfov_deg=head_cam.hfov_deg,
                    body_xz=body_xz,
                )
                self._coverage.stamp_self_cell(
                    sid, self._sim_t,
                    (float(body_pos.x), float(body_pos.y), float(body_pos.z)),
                )
            except Exception as exc:  # noqa: BLE001
                # Don't let a single tick's coverage glitch crash the
                # render loop; log once per second at most.
                now_sec = int(self._sim_t)
                if now_sec != getattr(self, "_last_cov_warn_sec", -1):
                    print(
                        f"[mumt-hitl] coverage update warn (spot{sid}): "
                        f"{exc!r}", flush=True,
                    )
                    self._last_cov_warn_sec = now_sec

            if sid < len(self._caption_workers):
                sec = self._coverage.coarse_label_for_world_xz(
                    body_xz[0], body_xz[1],
                )
                self._caption_workers[sid].post_observation(
                    rgb=rgb, t_sim=self._sim_t, sector=sec,
                    pose_x=body_xz[0],
                    pose_z=body_xz[1],
                    pose_yaw_rad=float(self._teleops[sid].state.yaw),
                    rgb_is_bgr=False,
                )

    def _refresh_state_snapshots(self) -> None:
        """Update the per-Spot dicts the AgentLoop reads via get_state."""
        if not self._state_snapshots or self._coverage is None:
            return
        cov_seen = int(
            (self._coverage.last_seen_t.max(axis=-1) > -1e8).sum()
        )
        cov_nav = int(self._coverage.is_navigable.sum())
        cov_pct = 100.0 * cov_seen / max(1, cov_nav)
        user_pos = self._humanoid_ao.translation
        user_xz = (float(user_pos.x), float(user_pos.z))
        user_sector = self._coverage.coarse_label_for_world_xz(
            user_xz[0], user_xz[1],
        )

        with self._state_lock:
            for sid in range(len(self._state_snapshots)):
                body_pos = self._teleops[sid].state.position
                body_xz = (float(body_pos.x), float(body_pos.z))
                sec = self._coverage.coarse_label_for_world_xz(
                    body_xz[0], body_xz[1],
                )
                running_str: Optional[str] = None
                if (
                    self._current_ctlrs[sid] is not None
                    and self._current_names[sid] is not None
                ):
                    age = (
                        time.monotonic() - self._current_started_t[sid]
                    )
                    running_str = (
                        f"{self._current_names[sid]} ({age:.1f}s elapsed)"
                    )
                snap = self._state_snapshots[sid]
                snap["pose_xz"] = body_xz
                snap["yaw_rad"] = float(self._teleops[sid].state.yaw)
                snap["sector"] = sec
                snap["sim_t"] = float(self._sim_t)
                snap["coverage_summary"] = (
                    f"shared coverage: {cov_seen}/{cov_nav} cells "
                    f"({cov_pct:.1f}%) seen"
                )
                snap["running_tool"] = running_str
                snap["user_pose_xz"] = user_xz
                snap["user_sector"] = user_sector

    # --- active-spot helper --------------------------------------------

    def _active_spot_id(self) -> Optional[int]:
        """Spot ID currently shown in the main HUD POV slot, or None.

        ``None`` when the map view is active (no spot is "the
        displayed one"); in that state manual override and per-Spot
        right-stick driving are no-ops.
        """
        if self._hud_mode == _HUD_MODE_SPOT0_POV:
            return 0
        if self._hud_mode == _HUD_MODE_SPOT1_POV:
            return 1
        return None

    # --- push-to-talk (B) + pointing (trigger) -------------------------

    def _handle_pointing(self, xr) -> None:
        """Update the latest pointer hit while the index trigger is
        held, and drive the visible laser line in the Unity client.

        Cheap to call every frame; raycasts hit habitat-sim's BVH which
        is fast enough at 60 fps.
        """
        try:
            held = xr.right_controller.get_button(
                XRButton.PRIMARY_INDEX_TRIGGER,
            )
        except Exception:  # noqa: BLE001 -- XRButton enum mismatch
            return
        if not held:
            self._latest_pointer_hit = None
            # Tell the client to hide the laser. PointerKeyframe
            # internally suppresses redundant release packets.
            self._pointer_kf.set_visible(False)
            return

        sample = self._sample_pointer_now()
        if sample is None:
            # Lost the controller pose this frame; keep the line
            # hidden rather than rendering a stale ray.
            self._pointer_kf.set_visible(False)
            return

        origin_xyz, endpoint_xyz, best_hit = sample
        if best_hit is not None:
            self._latest_pointer_hit = best_hit
        # Always show the line while the trigger is held -- even if
        # the ray missed every surface (the user gets a 25 m ray into
        # the void) so they have continuous visual feedback that
        # pointing is active.
        self._pointer_kf.set_visible(
            True,
            origin_world=origin_xyz,
            endpoint_world=endpoint_xyz,
            color_rgb=self._pointer_color,
        )

    def _handle_ptt(self, xr) -> None:
        """Right B-button push-to-talk state machine.

        Down-edge (B pressed): open the recorder. Up-edge (B released):
        stop, transcribe, and if a pointer hit was captured during the
        PTT window, prepend ``(user pointed to (x, y, z))`` to the
        transcript before posting to the orchestrator.
        """
        if self._ptt_recorder is None or self._stt is None:
            return
        try:
            down = xr.right_controller.get_button_down(XRButton.TWO)
            up = xr.right_controller.get_button_up(XRButton.TWO)
        except Exception:  # noqa: BLE001 -- XRButton enum mismatch
            return

        if down and not self._ptt_active:
            try:
                self._ptt_recorder.start()
            except Exception as exc:  # noqa: BLE001
                print(f"[mumt-hitl] PTT start failed: {exc}", flush=True)
                return
            self._ptt_active = True
            # Reset the per-PTT pointer cache; the trigger-driven
            # ``_handle_pointing`` will repopulate it while the trigger
            # is held during this PTT window.
            self._ptt_point_cache = []
            print("[mumt-hitl] PTT: recording", flush=True)
        elif up and self._ptt_active:
            wav = b""
            try:
                wav = self._ptt_recorder.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[mumt-hitl] PTT stop failed: {exc}", flush=True)
            self._ptt_active = False
            print(
                f"[mumt-hitl] PTT: stop ({len(wav)} bytes captured)",
                flush=True,
            )
            if wav:
                fut = self._stt.transcribe(wav)
                self._ptt_pending.append(fut)

    def _sample_pointer_now(self):
        """Cast the controller ray and return its visualization geometry.

        Returns ``(origin_world, endpoint_world, best_hit)`` where:

          * ``origin_world`` -- (x, y, z) habitat world coords of the
            controller (line start).
          * ``endpoint_world`` -- (x, y, z) habitat world coords of the
            line end. If the ray hit a surface or the HUD map, this is
            the closest hit; otherwise it's a fallback point
            ``_pointer_fallback_dist_m`` along the controller's
            forward axis.
          * ``best_hit`` -- the closest non-None :class:`PointHit`, or
            ``None`` if the ray missed everything.

        Also (preserves prior behavior) appends ``best_hit`` to
        ``self._ptt_point_cache`` when non-None so the PTT release path
        can pick up the freshest aim.

        Returns ``None`` if the right controller pose is unavailable.
        """
        from mumt_sim.agent.pointing import (  # noqa: PLC0415
            closest_point_target, controller_forward,
            raycast_hud_map, raycast_world,
        )
        remote = self._app_service.remote_client_state
        if remote is None:
            return None
        # ``get_hand_pose(user_index=0, hand_idx=1)`` returns the right
        # controller's pose already converted into habitat agent-space
        # (z+ forward, with the 180-degree-around-Y flip the head pose
        # uses). Same convention as the head, so transforming (0,0,1)
        # by the returned quaternion gives the controller's forward.
        try:
            hand_pos, hand_rot = remote.get_hand_pose(0, 1)
        except Exception:  # noqa: BLE001
            return None
        if hand_pos is None or hand_rot is None:
            return None

        sim = self._app_service.sim
        world_hit = None
        try:
            world_hit = raycast_world(
                sim, hand_pos, hand_rot,
                max_dist_m=self._pointer_fallback_dist_m,
            )
        except Exception:  # noqa: BLE001
            world_hit = None

        hud_hit = None
        if (
            self._hud_mode == _HUD_MODE_MAP
            and self._coverage is not None
            and self._displays.has("map")
        ):
            head_pos, head_rot = remote.get_head_pose(0)
            if head_pos is not None and head_rot is not None:
                try:
                    hud_hit = raycast_hud_map(
                        hand_pos=hand_pos,
                        hand_quat=hand_rot,
                        head_pos=head_pos,
                        head_rot=head_rot,
                        hud_offset_local=self._main_hud_layout.offset,
                        hud_size_m=self._main_hud_layout.size,
                        x_min=self._coverage.x_min,
                        x_max=self._coverage.x_max,
                        z_min=self._coverage.z_min,
                        z_max=self._coverage.z_max,
                        floor_y=float(self._humanoid_ao.translation.y),
                        max_dist_m=5.0,
                    )
                except Exception:  # noqa: BLE001
                    hud_hit = None

        best = closest_point_target(world_hit, hud_hit)
        if best is not None:
            self._ptt_point_cache.append(best)

        origin_xyz = (
            float(hand_pos[0]), float(hand_pos[1]), float(hand_pos[2]),
        )
        if best is not None:
            endpoint_xyz = (
                float(best.world_xyz[0]),
                float(best.world_xyz[1]),
                float(best.world_xyz[2]),
            )
        else:
            # No hit -- extend the laser into the void along the
            # controller's forward axis so the user still sees the line.
            fwd = controller_forward(hand_rot)
            d = float(self._pointer_fallback_dist_m)
            endpoint_xyz = (
                origin_xyz[0] + float(fwd.x) * d,
                origin_xyz[1] + float(fwd.y) * d,
                origin_xyz[2] + float(fwd.z) * d,
            )
        return origin_xyz, endpoint_xyz, best

    def _poll_pending_transcriptions(self) -> None:
        """Drain completed STT futures and forward to the orchestrator."""
        from mumt_sim.agent.pointing import (  # noqa: PLC0415
            format_point_annotation,
        )
        if not self._ptt_pending:
            return
        still_pending: List[Future] = []
        for fut in self._ptt_pending:
            if not fut.done():
                still_pending.append(fut)
                continue
            try:
                text = (fut.result() or "").strip()
            except Exception as exc:  # noqa: BLE001
                print(f"[mumt-hitl] STT failed: {exc}", flush=True)
                continue
            if not text:
                print("[mumt-hitl] PTT: (empty transcript)", flush=True)
                continue
            annotation = ""
            # Two sources for the pointer hit, in priority order:
            # 1) The most recent hit captured during this PTT window
            #    while the trigger was held (preferred -- "I just
            #    pointed at this and finished talking").
            # 2) Whatever the trigger is pointing at right now, in
            #    case the user is still holding the trigger after
            #    releasing B.
            point_hit = None
            if self._ptt_point_cache:
                point_hit = self._ptt_point_cache[-1]
            elif self._latest_pointer_hit is not None:
                point_hit = self._latest_pointer_hit
            if point_hit is not None:
                annotation = format_point_annotation(point_hit) + " "
            full = annotation + text
            print(f"\n[user] {full}", flush=True)
            if self._orchestrator is not None:
                self._orchestrator.post_user_message(full)
        self._ptt_pending = still_pending


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mumt_hitl",
)
def main(config):
    hitl_main(config, lambda app_service: AppStateMumt(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
