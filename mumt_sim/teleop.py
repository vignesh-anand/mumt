"""Continuous-velocity keyboard teleop of a single kinematic Spot.

Design (M2a):

- ``SpotTeleop`` owns mutable pose state for one Spot articulated object plus its
  pan/tilt head (``mumt_sim.pan_tilt.PanTiltHead``). Each ``step(dt, controls)``
  integrates velocities and pushes new transforms onto the AO and the head agent.
- The body's XZ translation is clamped to the navmesh via
  ``pathfinder.try_step``, which slides the spot along walls when it tries to
  walk through one. Yaw, pan, and tilt are unconstrained (tilt is hard-clamped to
  ``params.tilt_min`` / ``params.tilt_max`` to avoid the camera flipping over).
- This module deliberately knows nothing about pygame, OpenXR, or any specific
  input source. Drivers fill a ``TeleopInput`` and pass it to ``step``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import habitat_sim
import magnum as mn

from mumt_sim.pan_tilt import PanTiltHead


@dataclass
class TeleopParams:
    """Tunable rates / limits for keyboard teleop. Defaults aim at a calm
    walking pace; pass a custom instance to ``SpotTeleop`` to tweak."""

    forward_speed: float = 0.8     # m/s along body +X
    yaw_rate: float = math.radians(90.0)   # rad/s around world +Y
    pan_rate: float = math.radians(90.0)   # rad/s on the head's pan axis
    tilt_rate: float = math.radians(60.0)  # rad/s on the head's tilt axis
    tilt_min: float = math.radians(-60.0)
    tilt_max: float = math.radians(60.0)
    boost_factor: float = 2.0      # multiplier on linear + yaw when boost is held


@dataclass
class TeleopInput:
    """Frame-by-frame snapshot of which directional axes the user is asserting.

    Booleans are 'this axis is held *now*'. Drivers (e.g. pygame) build one of
    these per frame from their event/key state.
    """

    forward: bool = False      # W
    backward: bool = False     # S
    yaw_left: bool = False     # A
    yaw_right: bool = False    # D
    pan_left: bool = False     # left arrow
    pan_right: bool = False    # right arrow
    tilt_up: bool = False      # up arrow
    tilt_down: bool = False    # down arrow
    boost: bool = False        # shift
    reset: bool = False        # R (edge-triggered by caller; teleop just acts)


@dataclass
class TeleopState:
    """Mutable state pushed to habitat-sim each ``step``."""

    position: mn.Vector3 = field(default_factory=lambda: mn.Vector3())
    yaw: float = 0.0
    pan: float = 0.0
    tilt: float = 0.0


def _yaw_quat(yaw_rad: float) -> mn.Quaternion:
    return mn.Quaternion.rotation(mn.Rad(yaw_rad), mn.Vector3.y_axis())


class SpotTeleop:
    """Drive one Spot AO + its pan/tilt head from keyboard-style inputs.

    >>> teleop = SpotTeleop(sim, spot_ao, pan_tilt_head)
    >>> teleop.step(dt=1/60, controls=TeleopInput(forward=True))
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        body_ao,
        head: PanTiltHead,
        params: Optional[TeleopParams] = None,
    ) -> None:
        self.sim = sim
        self.body_ao = body_ao
        self.head = head
        self.params = params if params is not None else TeleopParams()

        # Pull initial pose from the AO so reset() can restore it.
        initial_pos = mn.Vector3(body_ao.translation)
        initial_yaw = float(body_ao.rotation.angle())
        # ``Quaternion.angle()`` always returns [0, pi]; reconstruct sign from
        # the rotation axis (we know ours is around +/-Y).
        if body_ao.rotation.axis().y < 0:
            initial_yaw = -initial_yaw

        self._initial = TeleopState(
            position=mn.Vector3(initial_pos),
            yaw=initial_yaw,
            pan=head.state.pan,
            tilt=head.state.tilt,
        )
        self.state = TeleopState(
            position=mn.Vector3(initial_pos),
            yaw=initial_yaw,
            pan=head.state.pan,
            tilt=head.state.tilt,
        )
        # Snap once into habitat-sim so head.sync() reflects current state.
        self._push()

    def reset(self) -> None:
        """Restore Spot 0 to its construction-time pose / pan / tilt."""
        self.state = TeleopState(
            position=mn.Vector3(self._initial.position),
            yaw=self._initial.yaw,
            pan=self._initial.pan,
            tilt=self._initial.tilt,
        )
        self._push()

    def step(self, dt: float, controls: TeleopInput) -> None:
        """Integrate one frame of input and write the result into habitat-sim."""
        if controls.reset:
            self.reset()
            return
        if dt <= 0.0:
            self._push()
            return

        boost = self.params.boost_factor if controls.boost else 1.0

        # 1) Body XZ translation, clamped against the navmesh.
        forward_axis = (1.0 if controls.forward else 0.0) - (1.0 if controls.backward else 0.0)
        if forward_axis != 0.0:
            speed = self.params.forward_speed * boost * forward_axis
            cy = math.cos(self.state.yaw)
            sy = math.sin(self.state.yaw)
            # Body +X in world coordinates after a +Y yaw (right-handed):
            #   R_y(yaw) * (1, 0, 0) = (cos yaw, 0, -sin yaw).
            forward_world = mn.Vector3(cy, 0.0, -sy)
            attempted = self.state.position + forward_world * (speed * dt)
            self.state.position = mn.Vector3(
                self.sim.pathfinder.try_step(self.state.position, attempted)
            )

        # 2) Yaw (no navmesh constraint).
        yaw_axis = (1.0 if controls.yaw_left else 0.0) - (1.0 if controls.yaw_right else 0.0)
        if yaw_axis != 0.0:
            self.state.yaw += self.params.yaw_rate * boost * yaw_axis * dt

        # 3) Pan (free, wraps).
        pan_axis = (1.0 if controls.pan_left else 0.0) - (1.0 if controls.pan_right else 0.0)
        if pan_axis != 0.0:
            self.state.pan += self.params.pan_rate * pan_axis * dt

        # 4) Tilt (clamped).
        tilt_axis = (1.0 if controls.tilt_up else 0.0) - (1.0 if controls.tilt_down else 0.0)
        if tilt_axis != 0.0:
            new_tilt = self.state.tilt + self.params.tilt_rate * tilt_axis * dt
            self.state.tilt = max(self.params.tilt_min, min(self.params.tilt_max, new_tilt))

        self._push()

    def _push(self) -> None:
        """Mirror ``self.state`` into the AO + the pan/tilt head."""
        self.body_ao.translation = self.state.position
        self.body_ao.rotation = _yaw_quat(self.state.yaw)
        self.head.set_pan_tilt(pan=self.state.pan, tilt=self.state.tilt)
        self.head.sync()
