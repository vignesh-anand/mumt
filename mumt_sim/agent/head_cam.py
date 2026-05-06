"""Per-Spot RGB + depth head sensors for the HITL-app autonomy stack.

Mirrors ``mumt_sim.scene._make_camera_agent``'s head-sensor configuration
but attaches the sensors to a Spot articulated object's root scene node
instead of a separate habitat-sim ``Agent``. Sensors are created via
``sim.create_sensor(spec, body_node)`` -- the same pattern
``mumt_sim.vr_displays.SpotPovDisplay`` uses, which works with the
SimDriver patch that enables texture loading.

Used by ``scripts/mumt_hitl_app.py`` to feed the autonomy stack's
``CoverageMap.update_from_depth``, the ``CaptionWorker``, and the
controller ``ControllerCtx`` without requiring the (non-HITL) two-Spot
sim layout in ``mumt_sim.scene.make_sim``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

import habitat_sim
import magnum as mn  # noqa: F401 -- kept for parity with other modules

from .coverage import _matrix4_to_numpy

# Body-local position of Spot's "head camera". Same defaults the HUD-only
# ``SpotPovDisplay`` uses (see mumt_sim/vr_displays.py:277): roughly
# 30 cm in front of and 50 cm above the body origin -- where Spot's
# stereo cameras live in the URDF. Coordinates are body-local
# (+X forward, +Y up, +Z right).
_DEFAULT_HEAD_OFFSET: Tuple[float, float, float] = (0.292, 0.50, 0.0)

# Orientation that rotates the sensor's natural -Z forward to align with
# Spot's body +X axis (the way the body is "looking" after a +Y yaw
# applied to the AO). Identical to ``SpotPovDisplay``.
_DEFAULT_HEAD_ORIENTATION_EULER_RAD: Tuple[float, float, float] = (
    0.0, -math.pi / 2.0, 0.0,
)

# Match ``mumt_sim.scene`` defaults so the same captioning / coverage
# numbers carry over from the standalone two-Spot scripts.
_DEFAULT_HFOV_DEG: float = 110.0
_DEFAULT_RES_HW: Tuple[int, int] = (480, 640)


class SpotHeadCam:
    """Co-located COLOR + DEPTH sensor pair attached to a Spot AO body.

    Both sensors share offset, orientation, HFOV, and resolution. Use
    :meth:`render_rgb` / :meth:`render_depth` to grab fresh observations
    each tick, and :meth:`cam_T_world` to feed the camera-to-world
    transform into ``CoverageMap.update_from_depth``.

    >>> head_cam = SpotHeadCam(sim, spot_ao, spot_id=0)
    >>> rgb = head_cam.render_rgb()
    >>> depth = head_cam.render_depth()
    >>> cov.update_from_depth(
    ...     spot_id=0, t_now=t, cam_T_world=head_cam.cam_T_world(),
    ...     depth=depth, hfov_deg=head_cam.hfov_deg,
    ... )
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        spot_ao,
        *,
        spot_id: int,
        size_hw: Tuple[int, int] = _DEFAULT_RES_HW,
        hfov_deg: float = _DEFAULT_HFOV_DEG,
        head_offset: Tuple[float, float, float] = _DEFAULT_HEAD_OFFSET,
        orientation_euler_rad: Tuple[float, float, float] = (
            _DEFAULT_HEAD_ORIENTATION_EULER_RAD
        ),
    ) -> None:
        self._sim = sim
        self._spot_ao = spot_ao
        self.spot_id = int(spot_id)
        self.size_hw: Tuple[int, int] = (int(size_hw[0]), int(size_hw[1]))
        self.hfov_deg: float = float(hfov_deg)
        self.hfov_rad: float = math.radians(self.hfov_deg)

        # Two sensors with matching extrinsics. Habitat-sim is happy
        # parking multiple sensors on the same scene node; both are
        # rendered each frame when ``draw_observation`` is called.
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = f"_mumt_headcam_{spot_id}_rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = list(self.size_hw)
        rgb_spec.hfov = self.hfov_deg
        rgb_spec.position = list(head_offset)
        rgb_spec.orientation = list(orientation_euler_rad)
        self._rgb_sensor = sim.create_sensor(
            rgb_spec, spot_ao.root_scene_node,
        )

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = f"_mumt_headcam_{spot_id}_depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = list(self.size_hw)
        depth_spec.hfov = self.hfov_deg
        depth_spec.position = list(head_offset)
        depth_spec.orientation = list(orientation_euler_rad)
        self._depth_sensor = sim.create_sensor(
            depth_spec, spot_ao.root_scene_node,
        )

    @property
    def color_sensor(self):
        """The underlying COLOR sensor. Lets a HUD ``SpotPovDisplay``
        share the same raster instead of creating a redundant sensor."""
        return self._rgb_sensor

    @property
    def depth_sensor(self):
        return self._depth_sensor

    def render_rgb(self) -> Optional[np.ndarray]:
        """Latest RGB observation as ``(H, W, 3)`` uint8, or None.

        None is returned only when the underlying sensor refuses to
        produce a frame (e.g. renderer not yet warm). Caller treats it
        as a transient skip and tries again next tick.
        """
        self._rgb_sensor.draw_observation()
        obs = self._rgb_sensor.get_observation()
        if obs is None:
            return None
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
        return None

    def render_depth(self) -> Optional[np.ndarray]:
        """Latest depth observation as ``(H, W)`` float32 metres."""
        self._depth_sensor.draw_observation()
        obs = self._depth_sensor.get_observation()
        if obs is None:
            return None
        return np.asarray(obs, dtype=np.float32)

    def cam_T_world(self) -> np.ndarray:
        """Camera-to-world 4x4 numpy matrix for the COLOR sensor's node.

        The two sensors share offset and orientation so the depth
        sensor's transform is identical (within sub-millimetre drift).
        ``CoverageMap.update_from_depth`` consumes this matrix.
        """
        T_mn = self._rgb_sensor.node.absolute_transformation()
        return _matrix4_to_numpy(T_mn)


__all__ = ["SpotHeadCam"]
