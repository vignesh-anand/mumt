"""Server-side raycast helpers for "user pointed at X" annotations.

Two surfaces a user might be pointing at:

1. The world (scene mesh): use ``habitat_sim.Simulator.cast_ray``.
2. The HUD top-down map quad: analytic plane intersection with the
   quad's local rectangle, then map UV -> world XZ via the navmesh
   AABB the map was rendered with.

Both helpers take the right-controller pose as it arrives from the
Unity client (already converted to habitat-frame coordinates by
``CoordinateSystem.ToHabitatVector / ToHabitatQuaternion``). Forward is
``+Z`` in habitat agent space (matches the rest of the HITL app's
gaze-direction convention).

The functions are pure-ish: they take the simulator only when the
caller wants the scene cast. Pointing is opt-in per call so a host
running with autonomy disabled doesn't pay for it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import habitat_sim
import magnum as mn


# Forward axis in habitat agent space (matches AppStateMumt.head_rot
# usage: forward = rot.transform_vector(mn.Vector3(0, 0, 1))).
_FORWARD_LOCAL: mn.Vector3 = mn.Vector3(0.0, 0.0, 1.0)


@dataclass
class PointHit:
    """A 3D world-space pointing target with provenance."""

    world_xyz: Tuple[float, float, float]
    distance_m: float
    source: str  # "world" or "hud_map"


def _to_mn_vec3(v) -> mn.Vector3:
    if isinstance(v, mn.Vector3):
        return v
    return mn.Vector3(float(v[0]), float(v[1]), float(v[2]))


def _to_mn_quat(q) -> mn.Quaternion:
    if isinstance(q, mn.Quaternion):
        return q
    # Habitat ordering on the wire: [w, x, y, z]; magnum's Quaternion
    # constructor takes (Vector3 vec, float scalar) where vec = (x,y,z).
    return mn.Quaternion(
        mn.Vector3(float(q[1]), float(q[2]), float(q[3])),
        float(q[0]),
    )


def controller_forward(hand_quat) -> mn.Vector3:
    """Habitat-frame unit forward vector for a controller pose.

    The Quest controller's natural pointing axis is its +Z after the
    Unity->habitat conversion the client applies (see
    ``CoordinateSystem.ToHabitatQuaternion``). Caller can also pass a
    ``magnum.Quaternion`` directly.
    """
    q = _to_mn_quat(hand_quat)
    return q.transform_vector(_FORWARD_LOCAL).normalized()


def raycast_world(
    sim: habitat_sim.Simulator,
    hand_pos,
    hand_quat,
    *,
    max_dist_m: float = 25.0,
) -> Optional[PointHit]:
    """Cast a ray from the controller into the scene mesh.

    Returns the closest hit point as a :class:`PointHit`, or None if
    the ray escaped the scene without hitting anything.
    """
    origin = _to_mn_vec3(hand_pos)
    direction = controller_forward(hand_quat)
    if direction.length() < 1e-6:
        return None

    ray = habitat_sim.geo.Ray(origin, direction)
    try:
        results = sim.cast_ray(ray, max_distance=float(max_dist_m))
    except TypeError:
        # Older habitat-sim signatures used a positional max_distance.
        results = sim.cast_ray(ray, float(max_dist_m))
    except Exception:  # noqa: BLE001 - never let a stray cast crash the loop
        return None

    hits = getattr(results, "hits", None) or []
    if not hits:
        return None
    # ``hits`` is sorted near->far by habitat-sim. Take the first.
    h = hits[0]
    pt = h.point
    return PointHit(
        world_xyz=(float(pt[0]), float(pt[1]), float(pt[2])),
        distance_m=float(h.ray_distance),
        source="world",
    )


def raycast_hud_map(
    *,
    hand_pos,
    hand_quat,
    head_pos,
    head_rot,
    hud_offset_local: Tuple[float, float, float],
    hud_size_m: Tuple[float, float],
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    floor_y: float,
    max_dist_m: float = 5.0,
) -> Optional[PointHit]:
    """Intersect the controller ray with the HUD's top-down map quad.

    The map quad is parented to the XR camera with ``hud_offset_local``
    in head-local space and faces back toward the user. Its area is
    ``hud_size_m`` (width, height) in metres. If the ray crosses that
    rectangle we map the (u, v) hit -> world (x, z) using the navmesh
    AABB the map was rendered with, and clamp ``y = floor_y``.

    Returns None if the ray misses the rectangle or runs past
    ``max_dist_m``.
    """
    origin = _to_mn_vec3(hand_pos)
    direction = controller_forward(hand_quat)
    if direction.length() < 1e-6:
        return None

    head_p = _to_mn_vec3(head_pos)
    head_q = _to_mn_quat(head_rot)

    # Quad center in world space.
    local_off = mn.Vector3(*[float(x) for x in hud_offset_local])
    quad_center = head_p + head_q.transform_vector(local_off)

    # Quad axes (local +X = right, +Y = up, +Z = forward). The quad is
    # rendered facing -Z in head-local space (i.e. faces the user); its
    # normal in world space is therefore the head's forward inverted.
    quad_right = head_q.transform_vector(mn.Vector3(1.0, 0.0, 0.0))
    quad_up = head_q.transform_vector(mn.Vector3(0.0, 1.0, 0.0))
    quad_normal = -head_q.transform_vector(mn.Vector3(0.0, 0.0, 1.0))

    # Ray-plane intersection. plane: dot(p - center, normal) = 0.
    denom = mn.math.dot(direction, quad_normal)
    if abs(denom) < 1e-4:
        return None  # ray ~parallel to the quad
    t = mn.math.dot(quad_center - origin, quad_normal) / denom
    if t <= 0.0 or t > max_dist_m:
        return None

    hit_world = origin + direction * t
    rel = hit_world - quad_center

    half_w = 0.5 * float(hud_size_m[0])
    half_h = 0.5 * float(hud_size_m[1])
    u = mn.math.dot(rel, quad_right)
    v = mn.math.dot(rel, quad_up)
    if abs(u) > half_w or abs(v) > half_h:
        return None  # missed the rectangle

    # Normalize to [0, 1] across the quad.
    u01 = (u + half_w) / (2.0 * half_w)
    # The map texture has (0,0) at top-left in image space, but we
    # render it upright on the quad (top of image = top of quad). v
    # increases upward in world; row 0 = top of texture = high v.
    v01 = 1.0 - (v + half_h) / (2.0 * half_h)

    # Translate to world XZ via the navmesh AABB the map was rendered
    # with. Map images have +Z down (south) -> bottom of image; +X right
    # (east) -> right of image. Match TopDownMapDisplay's convention.
    world_x = x_min + u01 * (x_max - x_min)
    world_z = z_min + v01 * (z_max - z_min)
    world_y = float(floor_y)

    return PointHit(
        world_xyz=(float(world_x), world_y, float(world_z)),
        distance_m=float(t),
        source="hud_map",
    )


def closest_point_target(
    *candidates: Optional[PointHit],
) -> Optional[PointHit]:
    """Return the nearest non-None :class:`PointHit`, or None."""
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    return min(valid, key=lambda h: h.distance_m)


def format_point_annotation(hit: PointHit) -> str:
    """Render a hit as the ``(user pointed to (x, y, z))`` prefix the
    orchestrator expects to see at the head of a user message."""
    x, y, z = hit.world_xyz
    return f"(user pointed to ({x:+.2f}, {y:+.2f}, {z:+.2f}))"


__all__ = [
    "PointHit",
    "controller_forward",
    "raycast_world",
    "raycast_hud_map",
    "closest_point_target",
    "format_point_annotation",
]
