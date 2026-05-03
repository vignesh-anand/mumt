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
    """

    sim: habitat_sim.Simulator
    spot_id: int
    teleop: SpotTeleop
    coverage: Optional[CoverageMap] = None
    latest_rgb: Optional[np.ndarray] = None
    latest_rgb_is_bgr: bool = False

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
