"""Atomic action primitives for the autonomy stack.

Each primitive is implemented as a step-able ``Controller``: the main
loop hands it ``(dt, ctx)`` once per tick, the controller produces a
continuous drive command via ``ctx.teleop.drive(...)`` and returns
``None`` while still running, then a :class:`PrimitiveResult` when it
finishes (or fails). This keeps the window responsive, lets two Spots
run primitives in parallel, and matches the eventual LLM tool-call
shape: a wrapper can ``start()`` and poll ``step()`` until done.

Primitives in this module:

- :class:`GotoController` -- drive the Spot from its current position
  to a goal XZ via a navmesh shortest path, using a pure-pursuit-style
  controller with yaw alignment.
- :class:`MoveController` -- body-frame relative motion: rotate by
  ``dyaw_rad``, then translate by ``(forward_m, lateral_m)`` in the
  new body frame. Useful as both a building block for higher-level
  primitives and a manual "nudge" tool.
- :class:`SearchSectorController` -- random-sample greedy set-cover
  viewpoint planner + tour executor. Goes to K viewpoints inside a
  named sector, asks Gemma a search-tuned caption at each, returns a
  structured :class:`SearchResult` to the caller.
- :class:`FindLabelController` -- same planner + tour executor, but
  fires the YOLOE Jetson server (via :class:`OnDemandDetector`) every
  N ticks looking for a target label. On the first hit it aborts the
  remaining tour, computes a yaw offset from the bbox center, and
  rotates the spot to face the target.

All motion is routed through ``SpotTeleop.drive``, so the navmesh
clamp, AO push, and HUD pose all stay consistent with the keyboard
teleop path.
"""
from __future__ import annotations

import concurrent.futures as _cf
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np

import habitat_sim

from mumt_sim.teleop import SpotTeleop

import itertools

from .coverage import CoverageMap
from .detection import Detection, DetectionResponse, OnDemandDetector
from .perception import (
    SEARCH_VIEWPOINT_PROMPT,
    OnDemandCaptioner,
    SearchViewpointCaption,
    parse_search_caption,
)
from .visibility import Viewpoint, plan_search_tour


def _order_tour_by_travel(
    start_xz: tuple[float, float],
    tour: list[Viewpoint],
    exact_max_k: int = 8,
) -> list[Viewpoint]:
    """Reorder ``tour`` to minimise total Euclidean travel from
    ``start_xz``, visiting each viewpoint exactly once.

    For ``len(tour) <= exact_max_k`` we brute-force every permutation
    (8! = 40320 paths, each scored in O(K) on tiny floats -- well
    under a millisecond). For larger K we fall back to nearest-
    neighbour greedy. Euclidean distance ignores walls, but for a
    5 m sector + 8 neighbours that's almost always fine; we can swap
    in geodesic distances later if needed.
    """
    n = len(tour)
    if n <= 1:
        return list(tour)

    coords = np.array([(vp.x, vp.z) for vp in tour], dtype=np.float64)
    start = np.array(start_xz, dtype=np.float64)

    def total_cost(perm: tuple[int, ...]) -> float:
        prev = start
        cost = 0.0
        for i in perm:
            cur = coords[i]
            cost += float(np.linalg.norm(cur - prev))
            prev = cur
        return cost

    if n <= exact_max_k:
        best_perm = min(
            itertools.permutations(range(n)), key=total_cost
        )
        return [tour[i] for i in best_perm]

    # NN heuristic for K > exact_max_k.
    remaining = list(range(n))
    perm: list[int] = []
    cur = start
    while remaining:
        nxt = min(
            remaining, key=lambda i: float(np.linalg.norm(coords[i] - cur))
        )
        perm.append(nxt)
        cur = coords[nxt]
        remaining.remove(nxt)
    return [tour[i] for i in perm]


PrimitiveStatus = Literal[
    "success", "unreachable", "blocked", "aborted", "timeout"
]


@dataclass
class PrimitiveResult:
    """Outcome of a finished controller. ``status`` tells the caller
    whether the primitive achieved its goal; ``reason`` is a short
    human-readable string useful for logging and the HUD. Pose is
    captured at completion."""

    primitive: str
    status: PrimitiveStatus
    reason: str
    t_elapsed_s: float
    final_pose: tuple[float, float, float]  # (x, z, yaw_rad)
    path_followed: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class ControllerCtx:
    """Per-tick environment a controller needs to read pose and emit
    drive commands. Built once per Spot and reused across primitives.

    ``latest_rgb`` is the most recent head-cam frame, refreshed by the
    main loop each tick. Controllers that need a frame snapshot (e.g.
    :class:`SearchSectorController` calling Gemma) read it directly.
    None until the first frame arrives.

    ``latest_rgb_is_bgr`` matches the channel order of the frame above
    (the teleop loop currently passes RGB straight from habitat-sim,
    so this defaults to False).

    ``latest_depth`` is the most recent head-cam depth image (float
    metres, ``[H, W]``), aligned pixel-for-pixel with ``latest_rgb``
    when both come from the same head sensor rig. None until the
    first frame arrives. Used by :class:`FindLabelController` to map
    a YOLOE bbox into a world-space approach point.

    ``latest_camera_hfov_rad`` is the horizontal FOV the head camera
    was rendered with -- needed alongside ``latest_depth`` to
    convert ``Z_cam`` perpendicular depth into a horizontal range to
    a target pixel.
    """

    sim: habitat_sim.Simulator
    spot_id: int
    teleop: SpotTeleop
    coverage: Optional[CoverageMap] = None
    latest_rgb: Optional[np.ndarray] = None
    latest_rgb_is_bgr: bool = False
    latest_depth: Optional[np.ndarray] = None
    latest_camera_hfov_rad: float = math.radians(110.0)

    @property
    def body_xz(self) -> tuple[float, float]:
        p = self.teleop.state.position
        return float(p.x), float(p.z)

    @property
    def yaw(self) -> float:
        return float(self.teleop.state.yaw)

    @property
    def world_y(self) -> float:
        return float(self.teleop.state.position.y)


class Controller(ABC):
    """Step-able controller. The main loop owns the lifecycle:

    1. construct with the goal,
    2. ``start(ctx)`` once,
    3. call ``step(dt, ctx)`` every tick until it returns a
       :class:`PrimitiveResult`,
    4. (optional) ``abort(reason)`` between ticks to stop early.

    Implementations must be cheap to construct and may not block in
    ``step``."""

    name: str = "primitive"

    def __init__(self, timeout_s: float = 60.0) -> None:
        self.timeout_s = float(timeout_s)
        self._t_start: Optional[float] = None
        self._aborted: Optional[str] = None
        self._path_followed: list[tuple[float, float]] = []

    def start(self, ctx: ControllerCtx) -> None:
        self._t_start = time.monotonic()
        self._path_followed = [ctx.body_xz]

    @abstractmethod
    def step(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        ...

    def abort(self, reason: str = "manual abort") -> None:
        self._aborted = reason

    def status_text(self) -> str:
        return self.name

    # -- helpers shared across controllers ---------------------------------

    def _elapsed(self) -> float:
        return 0.0 if self._t_start is None else time.monotonic() - self._t_start

    def _record(self, ctx: ControllerCtx) -> None:
        cur = ctx.body_xz
        if not self._path_followed or _xz_dist(cur, self._path_followed[-1]) > 0.05:
            self._path_followed.append(cur)

    def _finish(
        self,
        ctx: ControllerCtx,
        status: PrimitiveStatus,
        reason: str,
    ) -> PrimitiveResult:
        ctx.teleop.drive(0.0)  # ensure no residual command
        return PrimitiveResult(
            primitive=self.name,
            status=status,
            reason=reason,
            t_elapsed_s=self._elapsed(),
            final_pose=(*ctx.body_xz, ctx.yaw),
            path_followed=list(self._path_followed),
        )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _wrap_to_pi(a: float) -> float:
    """Wrap ``a`` (radians) into ``[-pi, pi]``."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _xz_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _heading_to(src_xz: Sequence[float], dst_xz: Sequence[float]) -> float:
    """World-yaw that points from ``src_xz`` to ``dst_xz``. Matches the
    teleop body-frame convention: yaw=0 means body +X is along world
    +X, and the world XZ forward vector for yaw is ``(cos, -sin)``."""
    dx = float(dst_xz[0]) - float(src_xz[0])
    dz = float(dst_xz[1]) - float(src_xz[1])
    # yaw such that (cos yaw, -sin yaw) = (dx, dz) / |.|
    return math.atan2(-dz, dx)


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


GotoTarget = Union[str, tuple[float, float], tuple[float, float, float]]


def resolve_goto_target(
    target: GotoTarget,
    coverage: Optional[CoverageMap] = None,
) -> tuple[float, float]:
    """Convert a primitive target spec to a world XZ.

    Accepted shapes:
    - ``"C2"`` -- coarse-grid sector label (requires ``coverage``).
    - ``(x, z)`` -- world XZ tuple.
    - ``(x, y, z)`` -- world XYZ tuple, Y is dropped (we plan on the
      navmesh and read Y back from there).

    Raises ``ValueError`` on unrecognised shapes / unknown sectors.
    """
    if isinstance(target, str):
        if coverage is None:
            raise ValueError(
                f"target {target!r} is a sector label but no CoverageMap "
                f"was provided to resolve it"
            )
        return coverage.world_xz_for_coarse_label(target)
    if isinstance(target, (tuple, list)):
        if len(target) == 2:
            return float(target[0]), float(target[1])
        if len(target) == 3:
            return float(target[0]), float(target[2])
    raise ValueError(
        f"goto target {target!r} must be a sector label, (x, z), or (x, y, z)"
    )


# ---------------------------------------------------------------------------
# GotoController
# ---------------------------------------------------------------------------


@dataclass
class GotoConfig:
    """Tunable knobs for :class:`GotoController`."""

    forward_speed_mps: float = 0.8
    yaw_rate_rps: float = math.radians(120.0)
    yaw_align_threshold_rad: float = math.radians(15.0)
    """If heading error exceeds this, turn-in-place; otherwise drive
    forward with a proportional yaw correction."""
    yaw_correction_gain: float = 1.5
    """Proportional gain on heading error while moving forward, in
    rad/s per rad."""
    look_ahead_m: float = 0.6
    """Pure-pursuit look-ahead radius. The next waypoint we steer
    toward is the first one further than this from the body."""
    waypoint_advance_m: float = 0.4
    """Once we are within this distance of a non-final waypoint,
    advance to the next one."""
    goal_tolerance_m: float = 0.5
    """Done when within this distance of the final waypoint."""
    stuck_distance_m: float = 0.05
    stuck_window_s: float = 2.0
    """If the body XZ has moved less than ``stuck_distance_m`` over
    ``stuck_window_s``, we declare the primitive blocked."""
    timeout_s: float = 60.0


class GotoController(Controller):
    """Drive to a world XZ via the navmesh.

    Construct with the target and an optional :class:`GotoConfig`;
    after the first :meth:`step`, the path is planned (pull from
    ``habitat_sim.ShortestPath``) and pure-pursuit kicks in.
    """

    name = "goto"

    def __init__(
        self,
        target_xz: tuple[float, float],
        cfg: Optional[GotoConfig] = None,
    ) -> None:
        super().__init__(timeout_s=(cfg.timeout_s if cfg else 60.0))
        self.cfg = cfg if cfg is not None else GotoConfig()
        self.target_xz = (float(target_xz[0]), float(target_xz[1]))
        self._waypoints: list[tuple[float, float]] = []
        self._wp_idx: int = 0
        self._stuck_anchor: Optional[tuple[float, float]] = None
        self._stuck_anchor_t: float = 0.0
        self._planned: bool = False

    def status_text(self) -> str:
        wp = f"{self._wp_idx + 1}/{len(self._waypoints)}" if self._waypoints else "-/-"
        tx, tz = self.target_xz
        return (
            f"GOTO -> ({tx:+.1f}, {tz:+.1f}) wp {wp} "
            f"t={self._elapsed():4.1f}s"
        )

    def _plan(self, ctx: ControllerCtx) -> bool:
        start_xyz = np.asarray(
            [ctx.body_xz[0], ctx.world_y, ctx.body_xz[1]], dtype=np.float32
        )
        end_xyz = np.asarray(
            [self.target_xz[0], ctx.world_y, self.target_xz[1]], dtype=np.float32
        )
        # Snap both onto the navmesh; pathfinder.find_path wants on-mesh
        # endpoints. snap_point picks the nearest navigable point.
        start_xyz = np.asarray(ctx.sim.pathfinder.snap_point(start_xyz),
                               dtype=np.float32)
        end_xyz = np.asarray(ctx.sim.pathfinder.snap_point(end_xyz),
                             dtype=np.float32)
        if not (np.all(np.isfinite(start_xyz)) and np.all(np.isfinite(end_xyz))):
            return False

        path = habitat_sim.ShortestPath()
        path.requested_start = start_xyz
        path.requested_end = end_xyz
        if not ctx.sim.pathfinder.find_path(path):
            return False

        # path.points is (start, ..., end); strip the start and convert
        # to XZ so we don't need Y bookkeeping during pursuit.
        pts = [np.asarray(p, dtype=np.float32) for p in path.points]
        if not pts:
            return False
        self._waypoints = [(float(p[0]), float(p[2])) for p in pts]
        # If the first waypoint is essentially where we are, drop it.
        if _xz_dist(self._waypoints[0], ctx.body_xz) < 0.05 and len(self._waypoints) > 1:
            self._waypoints.pop(0)
        self._wp_idx = 0
        self._planned = True
        return True

    def step(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        if self._t_start is None:
            self.start(ctx)

        if self._aborted is not None:
            return self._finish(ctx, "aborted", self._aborted)

        if not self._planned:
            ok = self._plan(ctx)
            if not ok or not self._waypoints:
                return self._finish(
                    ctx, "unreachable",
                    f"no navmesh path to ({self.target_xz[0]:+.2f}, "
                    f"{self.target_xz[1]:+.2f})",
                )
            self._stuck_anchor = ctx.body_xz
            self._stuck_anchor_t = self._elapsed()

        # Advance past intermediate waypoints we've already reached.
        while (
            self._wp_idx < len(self._waypoints) - 1
            and _xz_dist(ctx.body_xz, self._waypoints[self._wp_idx])
                < self.cfg.waypoint_advance_m
        ):
            self._wp_idx += 1

        # Pick the steering target: the first waypoint past the
        # look-ahead radius (else: the current waypoint).
        steer_idx = self._wp_idx
        for k in range(self._wp_idx, len(self._waypoints)):
            if _xz_dist(ctx.body_xz, self._waypoints[k]) >= self.cfg.look_ahead_m:
                steer_idx = k
                break
            steer_idx = k
        steer_target = self._waypoints[steer_idx]
        final_target = self._waypoints[-1]

        # Done?
        if _xz_dist(ctx.body_xz, final_target) <= self.cfg.goal_tolerance_m:
            return self._finish(
                ctx, "success",
                f"reached goal in {self._elapsed():.1f}s",
            )

        # Stuck?
        if (
            self._stuck_anchor is not None
            and (self._elapsed() - self._stuck_anchor_t) >= self.cfg.stuck_window_s
        ):
            if _xz_dist(ctx.body_xz, self._stuck_anchor) < self.cfg.stuck_distance_m:
                return self._finish(
                    ctx, "blocked",
                    f"no progress for {self.cfg.stuck_window_s:.0f}s "
                    f"(navmesh edge or collision)",
                )
            self._stuck_anchor = ctx.body_xz
            self._stuck_anchor_t = self._elapsed()

        # Timeout?
        if self._elapsed() >= self.timeout_s:
            return self._finish(
                ctx, "timeout",
                f"primitive exceeded {self.timeout_s:.0f}s",
            )

        # Drive toward the steer target.
        desired_yaw = _heading_to(ctx.body_xz, steer_target)
        yaw_err = _wrap_to_pi(desired_yaw - ctx.yaw)
        if abs(yaw_err) > self.cfg.yaw_align_threshold_rad:
            # Turn-in-place to face the target.
            yaw_rps = self.cfg.yaw_rate_rps * (1.0 if yaw_err > 0 else -1.0)
            ctx.teleop.drive(dt, forward_mps=0.0, yaw_rps=yaw_rps)
        else:
            yaw_rps = max(
                -self.cfg.yaw_rate_rps,
                min(self.cfg.yaw_rate_rps, self.cfg.yaw_correction_gain * yaw_err),
            )
            ctx.teleop.drive(
                dt,
                forward_mps=self.cfg.forward_speed_mps,
                yaw_rps=yaw_rps,
            )

        self._record(ctx)
        return None


# ---------------------------------------------------------------------------
# MoveController
# ---------------------------------------------------------------------------


@dataclass
class MoveConfig:
    forward_speed_mps: float = 0.6
    lateral_speed_mps: float = 0.4
    yaw_rate_rps: float = math.radians(90.0)
    yaw_tolerance_rad: float = math.radians(2.0)
    xz_tolerance_m: float = 0.05
    timeout_s: float = 20.0


class MoveController(Controller):
    """Body-frame relative motion: rotate by ``dyaw_rad``, then
    translate by ``(forward_m, lateral_m)`` in the new body frame.

    The order is intentional -- we yaw first so the translation goes
    where the agent expects after the rotation, mirroring how a human
    would issue the command ("turn 90 degrees, then walk 1 m forward").
    """

    name = "move"

    def __init__(
        self,
        forward_m: float = 0.0,
        lateral_m: float = 0.0,
        dyaw_rad: float = 0.0,
        cfg: Optional[MoveConfig] = None,
    ) -> None:
        super().__init__(timeout_s=(cfg.timeout_s if cfg else 20.0))
        self.cfg = cfg if cfg is not None else MoveConfig()
        self.forward_m = float(forward_m)
        self.lateral_m = float(lateral_m)
        self.dyaw_rad = float(dyaw_rad)
        self._goal_yaw: Optional[float] = None
        self._goal_xz: Optional[tuple[float, float]] = None
        self._phase: Literal["yaw", "translate"] = "yaw"
        self._initialised: bool = False

    def status_text(self) -> str:
        if not self._initialised:
            return f"MOVE init"
        return (
            f"MOVE phase={self._phase} fwd={self.forward_m:+.2f}m "
            f"lat={self.lateral_m:+.2f}m dyaw={math.degrees(self.dyaw_rad):+.0f}deg "
            f"t={self._elapsed():4.1f}s"
        )

    def _initialise(self, ctx: ControllerCtx) -> None:
        x0, z0 = ctx.body_xz
        yaw0 = ctx.yaw
        self._goal_yaw = yaw0 + self.dyaw_rad
        cy = math.cos(self._goal_yaw)
        sy = math.sin(self._goal_yaw)
        # Body +X in world XZ at goal yaw: (cos, -sin).
        # Body +Z (right side) in world XZ at goal yaw: (sin, cos).
        gx = x0 + cy * self.forward_m + sy * self.lateral_m
        gz = z0 - sy * self.forward_m + cy * self.lateral_m
        self._goal_xz = (gx, gz)
        self._phase = "yaw" if abs(self.dyaw_rad) > self.cfg.yaw_tolerance_rad else "translate"
        self._initialised = True

    def step(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        if self._t_start is None:
            self.start(ctx)

        if self._aborted is not None:
            return self._finish(ctx, "aborted", self._aborted)

        if not self._initialised:
            self._initialise(ctx)

        if self._elapsed() >= self.timeout_s:
            return self._finish(
                ctx, "timeout",
                f"primitive exceeded {self.timeout_s:.0f}s",
            )

        assert self._goal_yaw is not None and self._goal_xz is not None

        if self._phase == "yaw":
            yaw_err = _wrap_to_pi(self._goal_yaw - ctx.yaw)
            if abs(yaw_err) <= self.cfg.yaw_tolerance_rad:
                self._phase = "translate"
            else:
                yaw_rps = self.cfg.yaw_rate_rps * (1.0 if yaw_err > 0 else -1.0)
                # Cap step to not overshoot.
                max_step = abs(yaw_err) / max(dt, 1e-6)
                yaw_rps = math.copysign(min(abs(yaw_rps), max_step), yaw_rps)
                ctx.teleop.drive(dt, forward_mps=0.0, yaw_rps=yaw_rps)
                self._record(ctx)
                return None

        # translate phase
        gx, gz = self._goal_xz
        x, z = ctx.body_xz
        dist = _xz_dist((x, z), (gx, gz))
        if dist <= self.cfg.xz_tolerance_m:
            return self._finish(
                ctx, "success",
                f"reached relative goal in {self._elapsed():.1f}s",
            )

        # Express remaining (gx-x, gz-z) in body frame at current yaw
        # so we can split it across forward/lateral velocities.
        cy = math.cos(ctx.yaw)
        sy = math.sin(ctx.yaw)
        # World -> body: forward = world_dx*cos - world_dz*sin
        #                lateral = world_dx*sin + world_dz*cos
        wdx = gx - x
        wdz = gz - z
        body_fwd = wdx * cy - wdz * sy
        body_lat = wdx * sy + wdz * cy
        # Cap velocities, scale-preserving.
        v_fwd = max(-self.cfg.forward_speed_mps, min(self.cfg.forward_speed_mps, body_fwd / max(dt, 1e-6)))
        v_lat = max(-self.cfg.lateral_speed_mps, min(self.cfg.lateral_speed_mps, body_lat / max(dt, 1e-6)))

        ctx.teleop.drive(dt, forward_mps=v_fwd, lateral_mps=v_lat, yaw_rps=0.0)
        self._record(ctx)
        return None


# ---------------------------------------------------------------------------
# SearchSectorController
# ---------------------------------------------------------------------------


@dataclass
class SearchObservation:
    """One viewpoint's worth of evidence inside a :class:`SearchResult`.

    ``caption`` is the parsed Gemma response when the on-demand caption
    succeeded; ``error`` is set instead when goto / face / Gemma all
    failed for this viewpoint (we still record the viewpoint so the
    caller sees what the planner intended). Exactly one of ``caption``
    / ``error`` is set.
    """

    viewpoint: Viewpoint
    reached_pose: Optional[tuple[float, float, float]]
    caption: Optional[SearchViewpointCaption] = None
    error: Optional[str] = None
    t_caption_s: float = 0.0


@dataclass
class SearchResult(PrimitiveResult):
    """Specialised :class:`PrimitiveResult` for sector search.

    Inherits the generic status / reason / final pose, adds the planner
    output and per-viewpoint observations. Designed to be the structured
    return value the LLM agent will consume; the human HUD prints a
    one-line summary per viewpoint.
    """

    sector: str = ""
    n_viewpoints_planned: int = 0
    observations: list[SearchObservation] = field(default_factory=list)


@dataclass
class SearchSectorConfig:
    """Tunable knobs for :class:`SearchSectorController`. Defaults
    match the agreed plan."""

    n_positions: int = 40
    n_headings: int = 12
    k_max: int = 6
    min_gain_cells: int = 10
    hfov_rad: float = math.radians(70.0)
    max_range_m: float = 5.0
    n_los_rays: int = 180
    plan_timeout_s: float = 10.0
    """Cap on the background plan_search_tour call. The default
    n_positions=100 typically completes in ~3 s."""
    caption_timeout_s: float = 20.0
    """Cap on each per-viewpoint Gemma call; we move on after this.
    Tuned for Gemma 4 26B which routinely takes 5-15 s per call.
    NOTE: cancelling a future that has already started running does
    not abort the underlying HTTP call -- the request continues to
    consume an OnDemandCaptioner worker until it actually completes.
    Give the captioner ``max_workers >= 2`` so timed-out calls don't
    block subsequent submits."""
    overall_timeout_s: float = 240.0
    """Hard wall-clock cap on the whole primitive."""
    settle_after_face_s: float = 0.25
    """Brief pause after the face-yaw move so the head image stops
    motion-blurring before we snapshot it for Gemma."""


class SearchSectorController(Controller):
    """Plan + execute a sector search.

    Workflow (state machine):

    1. ``PLANNING`` -- offload :func:`plan_search_tour` to a worker
       thread (it's ~3 s of numpy on a 100-position sample). Wait
       non-blockingly via :class:`concurrent.futures.Future`.
    2. For each planned viewpoint k:
       a. ``GOTO_K`` -- spawn an internal :class:`GotoController` to
          ``(vp.x, vp.z)``. On any non-success, record an error for
          this viewpoint and skip to (a) for k+1.
       b. ``FACE_K`` -- :class:`MoveController` with ``dyaw =
          wrap(vp.yaw - current yaw)`` so the head looks the planned
          way.
       c. ``SETTLE_K`` -- short pause so the head-cam frame is stable.
       d. ``CAPTION_K`` -- ``ctx`` exposes the latest RGB; submit it
          to ``on_demand`` with :data:`SEARCH_VIEWPOINT_PROMPT` +
          :func:`parse_search_caption`. Store the future.
       e. ``WAIT_K`` -- poll the future. When ready, append a
          :class:`SearchObservation` to the result.
    3. Done -- return a :class:`SearchResult`.

    Errors at any phase do not abort the whole primitive; we record
    them and move on. Manual abort + the overall timeout do.
    """

    name = "search"

    _Phase = Literal[
        "init", "planning", "goto", "face", "settle", "submit", "drain", "done"
    ]

    def __init__(
        self,
        sector: str,
        on_demand: OnDemandCaptioner,
        cfg: Optional[SearchSectorConfig] = None,
    ) -> None:
        super().__init__(
            timeout_s=(cfg.overall_timeout_s if cfg else 240.0)
        )
        self.sector = str(sector)
        self.on_demand = on_demand
        self.cfg = cfg if cfg is not None else SearchSectorConfig()

        self._phase: SearchSectorController._Phase = "init"
        self._tour: list[Viewpoint] = []
        self._idx: int = 0
        # Background plan_search_tour future and its private executor.
        self._plan_executor: Optional[_cf.ThreadPoolExecutor] = None
        self._plan_future: Optional[_cf.Future] = None
        self._plan_t0: float = 0.0
        # Inner controllers for goto / face phases.
        self._goto: Optional[GotoController] = None
        self._face: Optional[MoveController] = None
        self._settle_t0: float = 0.0
        # Pipelined captions: per-viewpoint snapshots + futures + arrival
        # poses. _pending maps vp_idx -> (future, submit_monotonic_t) for
        # captions that have been submitted but not yet collected.
        # _observations_by_idx accumulates SearchObservations as futures
        # resolve (or skip on goto failure); we flatten into a tour-order
        # list at finish.
        self._pending: dict[int, tuple[_cf.Future, float]] = {}
        self._observations_by_idx: dict[int, SearchObservation] = {}
        self._reached_poses: list[Optional[tuple[float, float, float]]] = []
        self._drain_t0: float = 0.0

    # ----- Controller lifecycle -------------------------------------------

    def status_text(self) -> str:
        if self._phase == "init":
            return f"SEARCH {self.sector} init"
        if self._phase == "planning":
            elapsed = time.monotonic() - self._plan_t0
            return f"SEARCH {self.sector} planning ({elapsed:4.1f}s)"
        if self._phase == "drain":
            return (
                f"SEARCH {self.sector} drain {len(self._pending)} "
                f"pending caption(s) t={self._elapsed():4.1f}s"
            )
        if self._phase == "done":
            return f"SEARCH {self.sector} done ({len(self._observations_by_idx)} obs)"
        # k of K (visit phases)
        kk = f"vp {self._idx + 1}/{len(self._tour)}"
        sub = self._phase.upper()
        pending = f" pend={len(self._pending)}" if self._pending else ""
        return f"SEARCH {self.sector} {kk} phase={sub}{pending} t={self._elapsed():4.1f}s"

    def _build_result(
        self,
        ctx: ControllerCtx,
        status: PrimitiveStatus,
        reason: str,
    ) -> SearchResult:
        ctx.teleop.drive(0.0)
        # Cancel any leftover pending futures (best effort; running
        # calls keep going on their worker threads). Record them as
        # errors so the result reflects what we know now.
        for vp_idx, (fut, t0) in list(self._pending.items()):
            fut.cancel()
            elapsed = time.monotonic() - t0
            self._observations_by_idx.setdefault(
                vp_idx,
                SearchObservation(
                    viewpoint=self._tour[vp_idx],
                    reached_pose=self._reached_poses[vp_idx]
                    if vp_idx < len(self._reached_poses) else None,
                    caption=None,
                    error=f"caption: cancelled after {elapsed:.1f}s "
                          f"(primitive ended)",
                ),
            )
        self._pending.clear()
        # Flatten observations into tour visit order (NOT planner index
        # order, since we already TSP-reordered the tour).
        ordered = [
            self._observations_by_idx[i]
            for i in range(len(self._tour))
            if i in self._observations_by_idx
        ]
        return SearchResult(
            primitive=self.name,
            status=status,
            reason=reason,
            t_elapsed_s=self._elapsed(),
            final_pose=(*ctx.body_xz, ctx.yaw),
            path_followed=list(self._path_followed),
            sector=self.sector,
            n_viewpoints_planned=len(self._tour),
            observations=ordered,
        )

    # ----- main step ------------------------------------------------------

    def step(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        if self._t_start is None:
            self.start(ctx)

        if self._aborted is not None:
            self._cleanup_executors()
            return self._build_result(ctx, "aborted", self._aborted)

        if self._elapsed() >= self.timeout_s:
            self._cleanup_executors()
            return self._build_result(
                ctx, "timeout",
                f"search exceeded {self.timeout_s:.0f}s wall budget",
            )

        # Drain any captions that completed while we drove. Cheap
        # poll, runs every tick regardless of which visit phase we're
        # in, so observations land as soon as the API responds.
        self._collect_done_pending()

        # Dispatch on phase.
        if self._phase == "init":
            return self._enter_planning(ctx)
        if self._phase == "planning":
            return self._step_planning(ctx)
        if self._phase == "goto":
            return self._step_goto(dt, ctx)
        if self._phase == "face":
            return self._step_face(dt, ctx)
        if self._phase == "settle":
            return self._step_settle(ctx)
        if self._phase == "submit":
            return self._step_submit(ctx)
        if self._phase == "drain":
            return self._step_drain(ctx)
        if self._phase == "done":
            obs_count = len(self._observations_by_idx)
            return self._build_result(
                ctx, "success",
                f"searched {self.sector}: {obs_count}/{len(self._tour)} viewpoints captioned",
            )
        return None

    # ----- helpers --------------------------------------------------------

    def _cleanup_executors(self) -> None:
        if self._plan_executor is not None:
            self._plan_executor.shutdown(wait=False, cancel_futures=True)
            self._plan_executor = None
        # Caption futures are owned by self.on_demand; we don't shut it down.

    def _enter_planning(self, ctx: ControllerCtx) -> Optional[PrimitiveResult]:
        if ctx.coverage is None:
            return self._build_result(
                ctx, "unreachable",
                "search needs ctx.coverage but none was provided",
            )
        cov = ctx.coverage
        # Validate sector label up front so we fail fast on typos.
        try:
            cov.sector_fine_indices(self.sector)
        except ValueError as exc:
            return self._build_result(
                ctx, "unreachable", f"unknown sector {self.sector!r}: {exc}",
            )

        cfg = self.cfg
        self._plan_executor = _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"SearchPlan-{ctx.spot_id}"
        )
        self._plan_t0 = time.monotonic()
        self._plan_future = self._plan_executor.submit(
            plan_search_tour,
            cov,
            self.sector,
            n_positions=cfg.n_positions,
            n_headings=cfg.n_headings,
            k_max=cfg.k_max,
            min_gain_cells=cfg.min_gain_cells,
            hfov_rad=cfg.hfov_rad,
            max_range_m=cfg.max_range_m,
            n_los_rays=cfg.n_los_rays,
        )
        self._phase = "planning"
        return None

    def _step_planning(self, ctx: ControllerCtx) -> Optional[PrimitiveResult]:
        assert self._plan_future is not None
        if (time.monotonic() - self._plan_t0) > self.cfg.plan_timeout_s:
            self._cleanup_executors()
            return self._build_result(
                ctx, "timeout",
                f"planning exceeded {self.cfg.plan_timeout_s:.0f}s",
            )
        if not self._plan_future.done():
            return None
        try:
            tour = self._plan_future.result()
        except Exception as exc:
            self._cleanup_executors()
            return self._build_result(
                ctx, "unreachable",
                f"plan_search_tour raised {type(exc).__name__}: {exc}",
            )
        self._cleanup_executors()
        # Reorder the planner's coverage-greedy output into travel-
        # optimal order from where the spot is right now. For K <= 8
        # this is exact (brute-force open TSP); above that we fall
        # back to nearest-neighbour. Cuts total drive distance
        # significantly when the planner picks viewpoints scattered
        # around the sector edges.
        self._tour = _order_tour_by_travel(ctx.body_xz, list(tour))
        self._idx = 0
        self._reached_poses = [None] * len(self._tour)
        if not self._tour:
            return self._build_result(
                ctx, "success",
                f"sector {self.sector} has no useful viewpoints (empty tour)",
            )
        self._phase = "goto"
        self._goto = GotoController((self._tour[0].x, self._tour[0].z))
        return None

    def _advance(self) -> None:
        """Move on to the next viewpoint, or to the drain phase (then
        ``done``) if all viewpoints have been visited."""
        self._idx += 1
        self._goto = None
        self._face = None
        if self._idx >= len(self._tour):
            # All visits done; if we have outstanding captions, wait
            # for them; otherwise we're already done.
            if self._pending:
                self._phase = "drain"
                self._drain_t0 = time.monotonic()
            else:
                self._phase = "done"
        else:
            vp = self._tour[self._idx]
            self._phase = "goto"
            self._goto = GotoController((vp.x, vp.z))

    def _record_skipped(self, reason: str) -> None:
        """Record a failure for the CURRENT viewpoint (no Gemma call
        was made). Distinct from caption failures recorded during
        drain."""
        vp_idx = self._idx
        vp = self._tour[vp_idx]
        self._observations_by_idx[vp_idx] = SearchObservation(
            viewpoint=vp,
            reached_pose=self._reached_poses[vp_idx]
            if vp_idx < len(self._reached_poses) else None,
            caption=None,
            error=reason,
        )

    def _set_reached_pose(self, ctx: ControllerCtx) -> None:
        """Record the spot's current pose for the CURRENT viewpoint
        index. Called at every motion-phase transition so we always
        have a "best-effort arrival pose" even if a later phase
        fails."""
        pose = (*ctx.body_xz, ctx.yaw)
        if self._idx < len(self._reached_poses):
            self._reached_poses[self._idx] = pose

    def _step_goto(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        assert self._goto is not None
        res = self._goto.step(dt, ctx)
        if res is None:
            return None
        self._set_reached_pose(ctx)
        if res.status != "success":
            self._record_skipped(f"goto: {res.status} ({res.reason})")
            self._advance()
            return None
        # Phase transition: face the planned yaw.
        vp = self._tour[self._idx]
        dyaw = _wrap_to_pi(vp.yaw_rad - ctx.yaw)
        self._face = MoveController(dyaw_rad=dyaw)
        self._phase = "face"
        return None

    def _step_face(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        assert self._face is not None
        res = self._face.step(dt, ctx)
        if res is None:
            return None
        self._set_reached_pose(ctx)
        if res.status not in ("success", "timeout"):
            # Non-success but also not catastrophic: record and skip.
            self._record_skipped(f"face: {res.status} ({res.reason})")
            self._advance()
            return None
        self._settle_t0 = time.monotonic()
        self._phase = "settle"
        return None

    def _step_settle(self, ctx: ControllerCtx) -> Optional[PrimitiveResult]:
        if (time.monotonic() - self._settle_t0) < self.cfg.settle_after_face_s:
            ctx.teleop.drive(0.0)
            return None
        self._phase = "submit"
        return None

    def _step_submit(self, ctx: ControllerCtx) -> Optional[PrimitiveResult]:
        """Submit the caption call and IMMEDIATELY advance to the
        next viewpoint. The future is parked in ``self._pending`` and
        polled every tick by ``_collect_done_pending``."""
        ctx.teleop.drive(0.0)
        if ctx.latest_rgb is None:
            self._record_skipped("caption: no RGB frame available on ctx")
            self._advance()
            return None
        # Hand the live frame to Gemma. We deliberately don't copy it:
        # the teleop loop allocates a fresh ndarray each tick so the
        # OnDemandCaptioner thread holds a Python ref that keeps the
        # buffer alive until the JPEG encode is done.
        future = self.on_demand.submit(
            ctx.latest_rgb,
            SEARCH_VIEWPOINT_PROMPT,
            parse_search_caption,
            rgb_is_bgr=ctx.latest_rgb_is_bgr,
        )
        self._pending[self._idx] = (future, time.monotonic())
        self._advance()
        return None

    def _collect_done_pending(self) -> None:
        """Poll the pending-captions dict; for any future that has
        completed, drop a SearchObservation in. Called every tick (in
        every phase) so observations land as early as possible."""
        if not self._pending:
            return
        now = time.monotonic()
        done_idxs: list[int] = []
        for vp_idx, (fut, t0) in self._pending.items():
            elapsed = now - t0
            if fut.done():
                try:
                    cap = fut.result()
                    self._observations_by_idx[vp_idx] = SearchObservation(
                        viewpoint=self._tour[vp_idx],
                        reached_pose=self._reached_poses[vp_idx],
                        caption=cap,
                        error=None,
                        t_caption_s=elapsed,
                    )
                except Exception as exc:
                    self._observations_by_idx[vp_idx] = SearchObservation(
                        viewpoint=self._tour[vp_idx],
                        reached_pose=self._reached_poses[vp_idx],
                        caption=None,
                        error=f"caption: {type(exc).__name__}: {exc} "
                              f"(after {elapsed:.1f}s)",
                    )
                done_idxs.append(vp_idx)
            elif elapsed > self.cfg.caption_timeout_s:
                fut.cancel()
                self._observations_by_idx[vp_idx] = SearchObservation(
                    viewpoint=self._tour[vp_idx],
                    reached_pose=self._reached_poses[vp_idx],
                    caption=None,
                    error=f"caption: timeout after {elapsed:.1f}s "
                          f"(cap={self.cfg.caption_timeout_s:.1f}s)",
                )
                done_idxs.append(vp_idx)
        for vp_idx in done_idxs:
            self._pending.pop(vp_idx, None)

    def _step_drain(self, ctx: ControllerCtx) -> Optional[PrimitiveResult]:
        """All viewpoints visited; just wait for outstanding caption
        futures to finish (or time out). _collect_done_pending already
        ran this tick at the top of step()."""
        ctx.teleop.drive(0.0)
        if not self._pending:
            self._phase = "done"
        return None


# ---------------------------------------------------------------------------
# FindLabelController
# ---------------------------------------------------------------------------


@dataclass
class FindObservation:
    """A single YOLOE detection event during a find tour.

    ``pose_at_capture`` is the body pose the spot had at the instant
    we grabbed the RGB frame for this detection -- crucial for the
    centering math, since the spot might have driven somewhere else
    by the time the response comes back.

    ``detections`` is the full per-frame list (any class), not just
    matches against the target label, so the caller can audit the
    server's behaviour.
    """

    t_capture_s: float
    pose_at_capture: tuple[float, float, float]
    image_size_wh: tuple[int, int]
    detections: list[Detection]
    inference_ms: float = 0.0
    total_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class FindResult(PrimitiveResult):
    """Specialised :class:`PrimitiveResult` for find-label.

    ``found=True`` iff a detection of ``target_label`` above the
    confidence threshold was returned by the server. The yaw-centering
    move that follows can still time out -- ``centered=True`` means
    the spot finished rotating to face the bbox.

    ``best_detection`` is the highest-confidence target detection
    seen during the tour. ``best_detection_pose`` and
    ``best_detection_image_size_wh`` are the snap-time body pose and
    server-decoded image size for THAT detection (needed to undo the
    bbox -> world yaw conversion later if the caller wants to).
    """

    target_label: str = ""
    sector: str = ""
    found: bool = False
    centered: bool = False
    approached: bool = False
    n_viewpoints_planned: int = 0
    n_detections_run: int = 0
    n_detections_failed: int = 0
    observations: list[FindObservation] = field(default_factory=list)
    best_detection: Optional[Detection] = None
    best_detection_pose: Optional[tuple[float, float, float]] = None
    best_detection_image_size_wh: Optional[tuple[int, int]] = None
    target_world_xz: Optional[tuple[float, float]] = None
    """Estimated world XZ of the target, computed at the moment of
    the centering hit by combining bbox-derived heading with the
    depth at the bbox center."""
    approach_world_xz: Optional[tuple[float, float]] = None
    """Where the approach goto was sent. ``approach_distance_m`` back
    from ``target_world_xz`` along the spot->target line."""
    target_range_m: Optional[float] = None
    """Estimated horizontal range from the snap pose to the target."""


@dataclass
class FindLabelConfig:
    """Tunable knobs for :class:`FindLabelController`.

    The planner block mirrors :class:`SearchSectorConfig` because both
    primitives share :func:`plan_search_tour`. The detection block is
    new: cadence (``detect_every_ticks``), confidence floor, request
    timeout. ``camera_hfov_rad`` MUST match the head-cam HFOV used to
    render frames -- :data:`scripts/teleop_two_spots_with_coverage.py`
    pins this at 110 deg via ``SPOT_HEAD_HFOV_DEG``.
    """

    # planner -- shared with SearchSectorConfig
    n_positions: int = 40
    n_headings: int = 12
    k_max: int = 6
    min_gain_cells: int = 10
    plan_hfov_rad: float = math.radians(70.0)
    """HFOV used by the *planner's* visibility cone. Deliberately
    narrower than the actual camera HFOV so we plan viewpoints that
    keep the target near the image center (where YOLOE is most
    reliable)."""
    max_range_m: float = 5.0
    n_los_rays: int = 180
    plan_timeout_s: float = 10.0

    # detection cadence + thresholds
    detect_every_ticks: int = 5
    """Submit at most one YOLOE call every N ticks. At ~60 fps this
    is ~12 Hz, which matches the Jetson's ~10 Hz steady-state cap on
    YOLOE-26-L. We also serialise to one in-flight call per controller
    so a slow response naturally throttles us."""
    detect_conf_threshold: float = 0.35
    """Minimum confidence to count a target detection as a hit."""
    detect_timeout_s: float = 4.0
    """Per-call HTTP cap. The first call with a new class list eats
    the YOLOE text-encoder cost (~1.5 s) so this needs headroom."""
    detect_yoloe_imgsz: Optional[int] = None
    """``None`` lets the server use its compiled default (640)."""

    # approach + centering
    camera_hfov_rad: float = math.radians(110.0)
    center_yaw_tolerance_rad: float = math.radians(3.0)
    """After the approach goto finishes, we re-aim at the target. Skip
    the final rotate when the residual yaw error is already within
    this tolerance."""
    approach_distance_m: float = 1.0
    """Stand-off distance from the target along the spot->target line
    where the approach goto stops. Set to 0.0 to drive ON TOP of the
    target (don't, the navmesh will refuse). The depth-derived target
    range is used as-is when it is already smaller than this; in that
    case the spot just rotates to face."""
    approach_min_drive_m: float = 0.25
    """If the depth says we are already within
    ``approach_distance_m + approach_min_drive_m`` of the target, skip
    the goto and just rotate to face. Avoids tiny meaningless drives."""
    approach_depth_window_px: int = 5
    """Half-width of the median-filter window taken around the bbox
    center pixel when sampling depth. Robust to single-pixel zeros /
    NaNs at object edges."""

    # overall
    overall_timeout_s: float = 240.0


class FindLabelController(Controller):
    """Find a labelled object inside a sector by touring planned
    viewpoints with YOLOE running continuously.

    State machine:

    1. ``planning`` -- offload :func:`plan_search_tour` (background
       thread). Same as search.
    2. For each planned viewpoint k: ``goto`` -> ``face`` -> advance.
       No per-viewpoint settle / submit; detections fire on a tick
       cadence regardless of which sub-phase we're in.
    3. Continuously (every ``detect_every_ticks`` ticks, when no
       detection is in flight): snap ``ctx.latest_rgb`` + body pose,
       submit ``OnDemandDetector.submit(rgb, [target_label])``.
    4. Each tick: poll the in-flight future. On completion, parse
       detections, store in ``observations``. If a detection of
       ``target_label`` clears ``detect_conf_threshold``, switch to
       ``center`` (cancels any inner goto/face).
    5. ``center`` -- compute target world yaw from
       ``pose_at_capture.yaw + atan2(bbox_center_x_norm * tan(HFOV/2))``
       and spawn a :class:`MoveController` to rotate to it.
    6. ``done`` -- emit :class:`FindResult` with ``found`` /
       ``centered`` flags.

    If the tour completes without a hit, the result is ``found=False``
    with a non-error status (``success`` -- the search ran cleanly,
    just nothing matched).
    """

    name = "find"

    _Phase = Literal[
        "init", "planning", "goto", "face",
        "approach", "face_target", "done"
    ]

    def __init__(
        self,
        sector: str,
        target_label: str,
        on_demand: OnDemandDetector,
        cfg: Optional[FindLabelConfig] = None,
    ) -> None:
        super().__init__(timeout_s=(cfg.overall_timeout_s if cfg else 240.0))
        self.sector = str(sector)
        self.target_label = str(target_label).strip()
        if not self.target_label:
            raise ValueError("target_label must be a non-empty string")
        self.on_demand = on_demand
        self.cfg = cfg if cfg is not None else FindLabelConfig()

        self._phase: FindLabelController._Phase = "init"
        self._tour: list[Viewpoint] = []
        self._idx: int = 0
        # Background plan_search_tour future and its private executor.
        self._plan_executor: Optional[_cf.ThreadPoolExecutor] = None
        self._plan_future: Optional[_cf.Future] = None
        self._plan_t0: float = 0.0
        # Inner controllers for goto / face / approach / face_target.
        self._goto: Optional[GotoController] = None
        self._face: Optional[MoveController] = None
        self._approach_goto: Optional[GotoController] = None
        self._face_target_move: Optional[MoveController] = None
        # Detection cadence: tick counter + at-most-one-in-flight slot.
        # The pending tuple keeps everything we need to interpret the
        # response in the world frame the call was issued in:
        #   (future, submit_t, snap_pose, snap_depth, snap_camera_hfov)
        # snap_depth is a reference to the depth ndarray that was live
        # at submit time; the teleop loop allocates a fresh ndarray
        # each tick (via sim.get_sensor_observations) so holding the
        # reference keeps the buffer stable.
        self._ticks_since_detect: int = 10**9  # force first detect ASAP
        self._pending_detect: Optional[
            tuple[
                _cf.Future, float, tuple[float, float, float],
                Optional[np.ndarray], float,
            ]
        ] = None
        # Accumulators for FindResult.
        self._observations: list[FindObservation] = []
        self._n_failed: int = 0
        self._best_det: Optional[Detection] = None
        self._best_pose: Optional[tuple[float, float, float]] = None
        self._best_size_wh: Optional[tuple[int, int]] = None
        # Hit bookkeeping; populated when we transition into approach.
        self._found: bool = False
        self._centered: bool = False
        self._approached: bool = False
        self._target_world_xz: Optional[tuple[float, float]] = None
        self._approach_world_xz: Optional[tuple[float, float]] = None
        self._target_range_m: Optional[float] = None

    # ----- Controller lifecycle -------------------------------------------

    def status_text(self) -> str:
        if self._phase == "init":
            return f"FIND {self.target_label!r} in {self.sector} init"
        if self._phase == "planning":
            elapsed = time.monotonic() - self._plan_t0
            return f"FIND {self.target_label!r} in {self.sector} planning ({elapsed:4.1f}s)"
        if self._phase == "approach":
            r = self._target_range_m
            return (
                f"FIND {self.target_label!r} in {self.sector} approach "
                f"({(r if r is not None else 0.0):.1f}m -> "
                f"{self.cfg.approach_distance_m:.1f}m) "
                f"t={self._elapsed():4.1f}s"
            )
        if self._phase == "face_target":
            return (
                f"FIND {self.target_label!r} in {self.sector} face_target "
                f"t={self._elapsed():4.1f}s"
            )
        if self._phase == "done":
            return (
                f"FIND {self.target_label!r} in {self.sector} done "
                f"(found={self._found} approached={self._approached} "
                f"centered={self._centered})"
            )
        kk = f"vp {self._idx + 1}/{len(self._tour)}"
        sub = self._phase.upper()
        pend = " det-pend" if self._pending_detect is not None else ""
        return (
            f"FIND {self.target_label!r} in {self.sector} {kk} phase={sub}"
            f"{pend} dets={len(self._observations)} t={self._elapsed():4.1f}s"
        )

    def _build_result(
        self,
        ctx: ControllerCtx,
        status: PrimitiveStatus,
        reason: str,
    ) -> FindResult:
        ctx.teleop.drive(0.0)
        # Best-effort cancel of any pending HTTP call.
        if self._pending_detect is not None:
            self._pending_detect[0].cancel()
            self._pending_detect = None
        return FindResult(
            primitive=self.name,
            status=status,
            reason=reason,
            t_elapsed_s=self._elapsed(),
            final_pose=(*ctx.body_xz, ctx.yaw),
            path_followed=list(self._path_followed),
            target_label=self.target_label,
            sector=self.sector,
            found=self._found,
            centered=self._centered,
            approached=self._approached,
            n_viewpoints_planned=len(self._tour),
            n_detections_run=len(self._observations),
            n_detections_failed=self._n_failed,
            observations=list(self._observations),
            best_detection=self._best_det,
            best_detection_pose=self._best_pose,
            best_detection_image_size_wh=self._best_size_wh,
            target_world_xz=self._target_world_xz,
            approach_world_xz=self._approach_world_xz,
            target_range_m=self._target_range_m,
        )

    # ----- main step ------------------------------------------------------

    def step(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        if self._t_start is None:
            self.start(ctx)

        if self._aborted is not None:
            self._cleanup_executors()
            return self._build_result(ctx, "aborted", self._aborted)

        if self._elapsed() >= self.timeout_s:
            self._cleanup_executors()
            return self._build_result(
                ctx, "timeout",
                f"find exceeded {self.timeout_s:.0f}s wall budget",
            )

        # Detection cadence: poll then maybe submit. Skipped while
        # planning (no useful frames yet) and while approach/face_target
        # (we've already committed to a target). Allowed during the
        # tour's goto/face so we can pick up the target on the way.
        if self._phase in ("goto", "face"):
            hit = self._poll_detection()
            if hit is not None:
                # Convert hit into an approach plan and switch phase.
                self._begin_approach(ctx, hit)
            else:
                self._maybe_submit_detection(ctx)

        # Dispatch on phase.
        if self._phase == "init":
            return self._enter_planning(ctx)
        if self._phase == "planning":
            return self._step_planning(ctx)
        if self._phase == "goto":
            return self._step_goto(dt, ctx)
        if self._phase == "face":
            return self._step_face(dt, ctx)
        if self._phase == "approach":
            return self._step_approach(dt, ctx)
        if self._phase == "face_target":
            return self._step_face_target(dt, ctx)
        if self._phase == "done":
            if self._found:
                reason_bits = [
                    f"found {self.target_label!r}"
                    + (f" at conf={self._best_det.confidence:.2f}"
                       if self._best_det is not None else "")
                ]
                if self._target_range_m is not None:
                    reason_bits.append(
                        f"range_at_detect={self._target_range_m:.2f}m"
                    )
                if not self._approached:
                    reason_bits.append("approach incomplete")
                if not self._centered:
                    reason_bits.append("centering incomplete")
                reason = "; ".join(reason_bits)
            else:
                reason = (
                    f"toured {len(self._tour)} viewpoint(s), "
                    f"{self.target_label!r} not detected"
                )
            return self._build_result(ctx, "success", reason)
        return None

    # ----- planning -------------------------------------------------------

    def _cleanup_executors(self) -> None:
        if self._plan_executor is not None:
            self._plan_executor.shutdown(wait=False, cancel_futures=True)
            self._plan_executor = None

    def _enter_planning(self, ctx: ControllerCtx) -> Optional[PrimitiveResult]:
        if ctx.coverage is None:
            return self._build_result(
                ctx, "unreachable",
                "find needs ctx.coverage but none was provided",
            )
        cov = ctx.coverage
        try:
            cov.sector_fine_indices(self.sector)
        except ValueError as exc:
            return self._build_result(
                ctx, "unreachable", f"unknown sector {self.sector!r}: {exc}",
            )

        cfg = self.cfg
        self._plan_executor = _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"FindPlan-{ctx.spot_id}"
        )
        self._plan_t0 = time.monotonic()
        self._plan_future = self._plan_executor.submit(
            plan_search_tour,
            cov,
            self.sector,
            n_positions=cfg.n_positions,
            n_headings=cfg.n_headings,
            k_max=cfg.k_max,
            min_gain_cells=cfg.min_gain_cells,
            hfov_rad=cfg.plan_hfov_rad,
            max_range_m=cfg.max_range_m,
            n_los_rays=cfg.n_los_rays,
        )
        self._phase = "planning"
        return None

    def _step_planning(self, ctx: ControllerCtx) -> Optional[PrimitiveResult]:
        assert self._plan_future is not None
        if (time.monotonic() - self._plan_t0) > self.cfg.plan_timeout_s:
            self._cleanup_executors()
            return self._build_result(
                ctx, "timeout",
                f"planning exceeded {self.cfg.plan_timeout_s:.0f}s",
            )
        if not self._plan_future.done():
            return None
        try:
            tour = self._plan_future.result()
        except Exception as exc:
            self._cleanup_executors()
            return self._build_result(
                ctx, "unreachable",
                f"plan_search_tour raised {type(exc).__name__}: {exc}",
            )
        self._cleanup_executors()
        self._tour = _order_tour_by_travel(ctx.body_xz, list(tour))
        self._idx = 0
        if not self._tour:
            return self._build_result(
                ctx, "success",
                f"sector {self.sector} has no useful viewpoints (empty tour)",
            )
        self._phase = "goto"
        self._goto = GotoController((self._tour[0].x, self._tour[0].z))
        return None

    # ----- tour visit -----------------------------------------------------

    def _advance(self) -> None:
        """Advance to the next viewpoint, or to ``done`` if exhausted."""
        self._idx += 1
        self._goto = None
        self._face = None
        if self._idx >= len(self._tour):
            self._phase = "done"
        else:
            vp = self._tour[self._idx]
            self._phase = "goto"
            self._goto = GotoController((vp.x, vp.z))

    def _step_goto(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        assert self._goto is not None
        res = self._goto.step(dt, ctx)
        if res is None:
            return None
        if res.status != "success":
            # Skip this viewpoint; keep searching.
            self._advance()
            return None
        vp = self._tour[self._idx]
        dyaw = _wrap_to_pi(vp.yaw_rad - ctx.yaw)
        self._face = MoveController(dyaw_rad=dyaw)
        self._phase = "face"
        return None

    def _step_face(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        assert self._face is not None
        res = self._face.step(dt, ctx)
        if res is None:
            return None
        # Whether face succeeded or timed out, we move on -- detection
        # has been firing throughout, so we either picked the target
        # up or didn't.
        self._advance()
        return None

    # ----- detection cadence ---------------------------------------------

    def _maybe_submit_detection(self, ctx: ControllerCtx) -> None:
        """Tick the cadence counter; submit a detection if it's time
        and there's no in-flight call. Snap the body pose, the depth
        image, and the camera HFOV alongside the frame so when the
        response arrives we can map the bbox back to a world XZ via
        depth in the same frame the call was issued in."""
        if self._pending_detect is not None:
            return
        self._ticks_since_detect += 1
        if self._ticks_since_detect < int(self.cfg.detect_every_ticks):
            return
        if ctx.latest_rgb is None:
            return
        snap_pose = (*ctx.body_xz, ctx.yaw)
        snap_depth = ctx.latest_depth
        snap_hfov = float(ctx.latest_camera_hfov_rad)
        try:
            future = self.on_demand.submit(
                ctx.latest_rgb,
                [self.target_label],
                conf=self.cfg.detect_conf_threshold,
                imgsz=self.cfg.detect_yoloe_imgsz,
                rgb_is_bgr=ctx.latest_rgb_is_bgr,
            )
        except Exception as exc:  # executor shutdown, etc.
            self._n_failed += 1
            self._observations.append(
                FindObservation(
                    t_capture_s=self._elapsed(),
                    pose_at_capture=snap_pose,
                    image_size_wh=(0, 0),
                    detections=[],
                    error=f"submit raised {type(exc).__name__}: {exc}",
                )
            )
            self._ticks_since_detect = 0
            return
        self._pending_detect = (
            future, time.monotonic(), snap_pose, snap_depth, snap_hfov,
        )
        self._ticks_since_detect = 0

    def _poll_detection(
        self,
    ) -> Optional[
        tuple[
            Detection, tuple[float, float, float],
            tuple[int, int], Optional[np.ndarray], float,
        ]
    ]:
        """Poll the in-flight detection. Returns ``(detection,
        snap_pose, image_size_wh, snap_depth, snap_hfov_rad)`` for the
        best target hit ready to commit to; ``None`` if the future is
        still running, errored, timed out, or returned nothing of
        interest. Side effect: appends a :class:`FindObservation`
        whenever the future resolves (success or failure)."""
        if self._pending_detect is None:
            return None
        future, t0, snap_pose, snap_depth, snap_hfov = self._pending_detect
        elapsed = time.monotonic() - t0
        if not future.done() and elapsed < self.cfg.detect_timeout_s:
            return None
        if not future.done():
            future.cancel()
            self._observations.append(
                FindObservation(
                    t_capture_s=self._elapsed(),
                    pose_at_capture=snap_pose,
                    image_size_wh=(0, 0),
                    detections=[],
                    error=f"detect: timeout after {elapsed:.2f}s",
                )
            )
            self._n_failed += 1
            self._pending_detect = None
            return None

        # future is done
        self._pending_detect = None
        try:
            response: DetectionResponse = future.result()
        except Exception as exc:
            self._observations.append(
                FindObservation(
                    t_capture_s=self._elapsed(),
                    pose_at_capture=snap_pose,
                    image_size_wh=(0, 0),
                    detections=[],
                    error=f"detect: {type(exc).__name__}: {exc}",
                )
            )
            self._n_failed += 1
            return None

        obs = FindObservation(
            t_capture_s=self._elapsed(),
            pose_at_capture=snap_pose,
            image_size_wh=response.image_size_wh,
            detections=response.detections,
            inference_ms=response.inference_ms,
            total_ms=response.total_ms,
        )
        self._observations.append(obs)

        best = response.best_for_label(self.target_label)
        if best is None or best.confidence < float(self.cfg.detect_conf_threshold):
            return None
        # Track best-overall regardless of whether we approach on this
        # one (we always commit to the FIRST hit for simplicity).
        if (
            self._best_det is None
            or best.confidence > self._best_det.confidence
        ):
            self._best_det = best
            self._best_pose = snap_pose
            self._best_size_wh = response.image_size_wh
        return (best, snap_pose, response.image_size_wh, snap_depth, snap_hfov)

    # ----- approach + centering -------------------------------------------

    @staticmethod
    def _sample_depth_at_pixel(
        depth: np.ndarray,
        cx_img: float,
        cy_img: float,
        img_w: int,
        img_h: int,
        win: int,
    ) -> Optional[float]:
        """Median-filter the depth value at the bbox center, with
        bbox pixels rescaled to the depth array's native shape (the
        two are 1:1 for our head sensors but we rescale anyway in case
        a future sensor pipes a different resolution into YOLOE)."""
        if depth is None or depth.size == 0 or img_w <= 0 or img_h <= 0:
            return None
        dh, dw = int(depth.shape[0]), int(depth.shape[1])
        px = int(round(float(cx_img) * dw / float(img_w)))
        py = int(round(float(cy_img) * dh / float(img_h)))
        px = max(0, min(dw - 1, px))
        py = max(0, min(dh - 1, py))
        w = max(0, int(win))
        y0, y1 = max(0, py - w), min(dh, py + w + 1)
        x0, x1 = max(0, px - w), min(dw, px + w + 1)
        patch = depth[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        finite = patch[np.isfinite(patch) & (patch > 0.0)]
        if finite.size == 0:
            return None
        return float(np.median(finite))

    def _begin_approach(
        self,
        ctx: ControllerCtx,
        hit: tuple[
            Detection, tuple[float, float, float], tuple[int, int],
            Optional[np.ndarray], float,
        ],
    ) -> None:
        """Commit to the first detection: convert bbox center +
        depth into a world target XZ, place an approach point
        ``approach_distance_m`` back from it along the spot->target
        line, and spawn a :class:`GotoController` to drive there.

        Falls back to a pure rotate-to-face-target when depth is
        missing / unusable, or when the spot is already inside the
        approach band (``approach_distance_m + approach_min_drive_m``).
        """
        det, snap_pose, (img_w, img_h), snap_depth, snap_hfov = hit
        self._found = True

        # Tear down whichever inner controller was running and drop
        # any pending detection -- we're committing to this hit.
        self._goto = None
        self._face = None
        if self._pending_detect is not None:
            self._pending_detect[0].cancel()
            self._pending_detect = None

        if img_w <= 0:
            # Pathological response; can't center on it.
            self._phase = "done"
            return

        # ----- bbox center -> world heading ------------------------------
        cx, cy = det.center_xy
        u_norm = (cx - 0.5 * img_w) / (0.5 * img_w)
        u_norm = max(-1.0, min(1.0, u_norm))
        # Sign trace (verified end-to-end against habitat-sim's own
        # camera projection in scripts/find_centering_smoketest.py):
        # PanTiltHead's -pi/2 Y-rotation maps camera image-right
        # (cam +X) to world +Z when body yaw=0. In this codebase
        # forward(yaw) = (cos yaw, -sin yaw) so increasing yaw turns
        # LEFT. Image-right -> body's right -> turn right -> yaw
        # decreases -> offset enters with a minus sign.
        half_hfov = 0.5 * float(snap_hfov)
        cam_offset_rad = math.atan(u_norm * math.tan(half_hfov))
        target_world_yaw = snap_pose[2] - cam_offset_rad

        # ----- bbox center + depth -> world target XZ --------------------
        # Habitat depth sensor returns Z_cam (perpendicular plane
        # distance). For a pixel at horizontal angle theta off the
        # optical axis, the horizontal Euclidean range to the surface
        # is z_cam / cos(theta). We ignore the small vertical tilt
        # (head is at INITIAL_SPOT_TILT ~ 10 deg) -- it inflates range
        # by less than 2% and we'd need to also know the head pitch
        # at snap time to correct it cleanly.
        z_cam = self._sample_depth_at_pixel(
            snap_depth, cx, cy, img_w, img_h,
            win=int(self.cfg.approach_depth_window_px),
        )

        snap_x, snap_z, _ = snap_pose
        forward = (math.cos(target_world_yaw), -math.sin(target_world_yaw))
        if z_cam is not None and z_cam > 0.0:
            target_range = z_cam / max(0.05, math.cos(cam_offset_rad))
            self._target_range_m = float(target_range)
            target_x = snap_x + target_range * forward[0]
            target_z = snap_z + target_range * forward[1]
            self._target_world_xz = (target_x, target_z)
        else:
            # No depth -> can't compute target XZ, fall back to rotate.
            self._target_range_m = None
            self._target_world_xz = None

        # ----- decide approach vs rotate-only ---------------------------
        approach_d = float(self.cfg.approach_distance_m)
        min_drive = float(self.cfg.approach_min_drive_m)

        if (
            self._target_world_xz is not None
            and self._target_range_m is not None
            and self._target_range_m > approach_d + min_drive
        ):
            drive_dist = self._target_range_m - approach_d
            approach_x = snap_x + drive_dist * forward[0]
            approach_z = snap_z + drive_dist * forward[1]
            self._approach_world_xz = (approach_x, approach_z)
            self._approach_goto = GotoController((approach_x, approach_z))
            self._phase = "approach"
            return

        # Already inside the approach band, or no depth -- skip the
        # goto and just rotate to face. We mark approached=True if we
        # know we're already within range.
        if self._target_range_m is not None and self._target_range_m <= approach_d + min_drive:
            self._approached = True
            self._approach_world_xz = (snap_x, snap_z)

        dyaw_now = _wrap_to_pi(target_world_yaw - ctx.yaw)
        self._face_target_move = MoveController(dyaw_rad=dyaw_now)
        self._phase = "face_target"

    def _step_approach(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        assert self._approach_goto is not None
        res = self._approach_goto.step(dt, ctx)
        if res is None:
            return None
        self._approached = res.status == "success"
        # Hand off to the final yaw alignment. After Goto the spot is
        # roughly facing the goal; recomputing the heading from the
        # goal's location to the (now stale) target world XZ gives us
        # the clean final aim.
        if self._target_world_xz is not None:
            desired_yaw = _heading_to(ctx.body_xz, self._target_world_xz)
            dyaw = _wrap_to_pi(desired_yaw - ctx.yaw)
            if abs(dyaw) > self.cfg.center_yaw_tolerance_rad:
                self._face_target_move = MoveController(dyaw_rad=dyaw)
                self._phase = "face_target"
                return None
            # Already within tolerance -- mark centered and finish.
            self._centered = True
        self._phase = "done"
        return None

    def _step_face_target(
        self, dt: float, ctx: ControllerCtx
    ) -> Optional[PrimitiveResult]:
        assert self._face_target_move is not None
        res = self._face_target_move.step(dt, ctx)
        if res is None:
            return None
        self._centered = res.status == "success"
        self._phase = "done"
        return None
