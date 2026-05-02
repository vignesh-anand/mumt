"""Script-owned pan/tilt heads bound to habitat-sim camera agents.

The design (M1):

- Each Spot has its own habitat-sim Agent (a camera rig) -- a separate concept from
  the visual Spot articulated object. The agent's body holds RGB+depth sensors.
- We keep the agent's pose in our script state (a ``HeadState`` dataclass),
  *not* in habitat-lab. ``PanTiltHead.sync()`` is the one method that pushes
  that state to habitat-sim each frame.
- Yaw composition: ``agent_body_rotation = body_yaw_world * pan_quat``.
  The body holds (body yaw + pan); tilt is applied locally to the sensor scene
  node so we never accumulate roll.
- Translation: agent body sits at ``body_world * head_offset`` (transform_point).

This module deliberately does not import habitat-lab.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import habitat_sim
import magnum as mn

from mumt_sim.agents import SPOT_HEAD_OFFSET


@dataclass
class HeadState:
    """All the state needed to position a Spot's pan/tilt head."""

    pan: float = 0.0
    """Yaw, radians, around world up. Composed *after* the body's own yaw."""

    tilt: float = 0.0
    """Pitch, radians, around the body's local X. Applied to the sensor node."""

    head_offset: mn.Vector3 = field(
        default_factory=lambda: mn.Vector3(SPOT_HEAD_OFFSET)
    )
    """Offset from the body's origin to the camera, expressed in body frame."""


def _quat_from_matrix3(m: mn.Matrix3) -> mn.Quaternion:
    """Build a magnum.Quaternion from a magnum.Matrix3 rotation."""
    # Magnum >= 2020.06 exposes Quaternion.from_matrix; fall back to constructor.
    if hasattr(mn.Quaternion, "from_matrix"):
        return mn.Quaternion.from_matrix(m)
    return mn.Quaternion.from_matrix(m)  # type: ignore[attr-defined]


class PanTiltHead:
    """One pan/tilt head bound to (a) a habitat-sim Agent and (b) a body articulated object.

    Lifecycle:

    >>> head = PanTiltHead(sim, agent_id=1, body_ao=spot_ao)
    >>> head.set_pan_tilt(pan=0.5, tilt=-0.1)
    >>> head.sync()                    # push state into habitat-sim
    >>> obs = sim.get_sensor_observations()
    >>> rgb = obs[1]["head_rgb"]
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        agent_id: int,
        body_ao,
        state: Optional[HeadState] = None,
    ) -> None:
        self.sim = sim
        self.agent_id = agent_id
        self.body_ao = body_ao
        self.state = state if state is not None else HeadState()

    def set_pan_tilt(self, pan: float, tilt: float) -> None:
        """Replace pan/tilt in radians."""
        self.state.pan = float(pan)
        self.state.tilt = float(tilt)

    # Quaternion that re-maps the habitat-sim camera's local -Z axis (default
    # forward) onto the Spot URDF's local +X axis (body forward / snout). Without
    # this the head camera ends up pointing ~90 deg to the Spot's right.
    _BODY_TO_CAMERA_BASIS = mn.Quaternion.rotation(
        mn.Rad(-math.pi / 2.0), mn.Vector3.y_axis()
    )

    def sync(self) -> None:
        """Push current state into habitat-sim. Call before ``get_sensor_observations``."""
        # 1) compute world transforms from the Spot AO
        T_body: mn.Matrix4 = self.body_ao.transformation
        head_world = T_body.transform_point(self.state.head_offset)

        body_q = _quat_from_matrix3(T_body.rotation())
        pan_q = mn.Quaternion.rotation(mn.Rad(self.state.pan), mn.Vector3.y_axis())
        body_plus_pan_q = body_q * pan_q * self._BODY_TO_CAMERA_BASIS

        # 2) push position + (body_yaw + pan) onto the agent body
        agent = self.sim.get_agent(self.agent_id)
        st = agent.get_state()
        st.position = [head_world.x, head_world.y, head_world.z]
        # habitat-sim's AgentState expects an (x, y, z, w) coeff list, not a
        # magnum.Quaternion (which isn't subscriptable).
        v = body_plus_pan_q.vector
        st.rotation = [v.x, v.y, v.z, body_plus_pan_q.scalar]
        agent.set_state(st)

        # 3) push tilt as a local pitch on every sensor node hanging off this agent
        tilt_q = mn.Quaternion.rotation(mn.Rad(self.state.tilt), mn.Vector3.x_axis())
        for sensor in agent._sensors.values():
            sensor.node.rotation = tilt_q
