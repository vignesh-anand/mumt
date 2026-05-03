"""Atomic action primitives for the autonomy stack.

Each primitive is implemented as a step-able ``Controller``: the main
loop hands it ``(dt, ctx)`` once per tick, the controller produces a
continuous drive command via ``ctx.teleop.drive(...)`` and returns
``None`` while still running, then a :class:`PrimitiveResult` when it
finishes (or fails). This keeps the window responsive, lets two Spots
run primitives in parallel, and matches the eventual LLM tool-call
shape: a wrapper can ``start()`` and poll ``step()`` until done.

The two primitives in this module:

- :class:`GotoController` -- drive the Spot from its current position
  to a goal XZ via a navmesh shortest path, using a pure-pursuit-style
  controller with yaw alignment.
- :class:`MoveController` -- body-frame relative motion: rotate by
  ``dyaw_rad``, then translate by ``(forward_m, lateral_m)`` in the
  new body frame. Useful as both a building block for higher-level
  primitives and a manual "nudge" tool.

All motion is routed through ``SpotTeleop.drive``, so the navmesh
clamp, AO push, and HUD pose all stay consistent with the keyboard
teleop path.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Union

import numpy as np

import habitat_sim

from mumt_sim.teleop import SpotTeleop

from .coverage import CoverageMap


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
    drive commands. Built once per Spot and reused across primitives."""

    sim: habitat_sim.Simulator
    spot_id: int
    teleop: SpotTeleop
    coverage: Optional[CoverageMap] = None

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
