"""Navmesh-aware spawn utilities."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def sample_navmesh_points(
    pathfinder,
    n: int,
    min_sep: float = 2.0,
    max_tries: int = 200,
    rng: "np.random.Generator | None" = None,
) -> List[Sequence[float]]:
    """Sample ``n`` points on the navmesh, each at least ``min_sep`` metres apart.

    The pathfinder draws from its own internal RNG inside ``get_random_navigable_point``;
    ``rng`` is unused for that call but kept in the signature for future deterministic
    sampling implementations. ``max_tries`` is a soft cap on attempts.

    Raises:
        RuntimeError: if we cannot find ``n`` adequately-spaced points within ``max_tries``.
    """
    if not pathfinder.is_loaded:
        raise RuntimeError(
            "pathfinder.is_loaded == False; the loaded scene has no navmesh"
        )

    pts: List[Sequence[float]] = []
    for _ in range(max_tries):
        p = pathfinder.get_random_navigable_point()
        # is_navigable() already implied by get_random_navigable_point(), but defend
        # against the (-Inf,-Inf,-Inf) sentinel that older habitat-sim returns when
        # the navmesh is empty.
        if not np.all(np.isfinite(p)):
            continue

        far_enough = all(np.linalg.norm(np.asarray(p) - np.asarray(q)) > min_sep for q in pts)
        if far_enough:
            pts.append(p)
            if len(pts) == n:
                return pts

    raise RuntimeError(
        f"Could not sample {n} navmesh points with min_sep={min_sep} "
        f"after {max_tries} tries (got {len(pts)})"
    )


def _geodesic_close(pathfinder, a, b, slack: float = 1.5) -> bool:
    """True if the navmesh-geodesic distance from ``a`` to ``b`` is at most
    ``slack`` x the straight-line distance. This filters out point pairs that
    are close in 3D but separated by walls (which would force a long detour
    around a doorway)."""
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    if not pathfinder.find_path(path):
        return False
    straight = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    if straight < 1e-3:
        return True
    return path.geodesic_distance <= straight * slack


def sample_navmesh_cluster(
    pathfinder,
    n: int,
    min_sep: float = 1.0,
    cluster_radius: float = 2.5,
    max_tries: int = 2000,
    same_room_slack: float = 1.4,
) -> List[Sequence[float]]:
    """Sample ``n`` navigable points all within ``cluster_radius`` metres of the
    first one, at least ``min_sep`` metres from each other, and with no walls
    between them (geodesic distance / straight-line distance <= ``same_room_slack``).

    Use this for the M1 group-shot render where we want all agents on screen at
    once and able to see each other. The pathfinder is used both for the anchor
    draw and for snapping candidate offsets via ``snap_point``.
    """
    if not pathfinder.is_loaded:
        raise RuntimeError(
            "pathfinder.is_loaded == False; the loaded scene has no navmesh"
        )

    rng = np.random.default_rng()

    for _outer in range(64):  # restart anchor a few times if cluster fails
        anchor = pathfinder.get_random_navigable_point()
        if not np.all(np.isfinite(anchor)):
            continue
        pts: List[Sequence[float]] = [anchor]
        for _ in range(max_tries):
            theta = rng.uniform(0.0, 2 * np.pi)
            r = rng.uniform(min_sep, cluster_radius)
            cand_xyz = np.asarray(anchor, dtype=np.float32).copy()
            cand_xyz[0] += float(r * np.cos(theta))
            cand_xyz[2] += float(r * np.sin(theta))
            snapped = pathfinder.snap_point(cand_xyz)
            snapped = np.asarray(snapped)
            if not np.all(np.isfinite(snapped)):
                continue
            if np.linalg.norm(snapped[[0, 2]] - np.asarray(anchor)[[0, 2]]) > cluster_radius:
                continue
            if not all(
                np.linalg.norm(snapped - np.asarray(q)) > min_sep for q in pts
            ):
                continue
            if not all(
                _geodesic_close(pathfinder, snapped, q, slack=same_room_slack)
                for q in pts
            ):
                continue
            pts.append(snapped.tolist())
            if len(pts) == n:
                return pts

    raise RuntimeError(
        f"Could not cluster {n} navmesh points within radius={cluster_radius}, "
        f"min_sep={min_sep} after retries"
    )


def find_open_spawn_spot(
    pathfinder,
    min_clearance: float = 1.5,
    n_samples: int = 800,
) -> Tuple[Sequence[float], float]:
    """Search the navmesh for the most-open navigable point.

    Returns ``(point, clearance)`` where ``clearance`` is the distance in
    metres from ``point`` to the nearest obstacle (wall or static collider).
    Samples ``n_samples`` random navigable points and keeps the best.

    Raises RuntimeError if no sample meets ``min_clearance``.
    """
    if not pathfinder.is_loaded:
        raise RuntimeError("pathfinder.is_loaded == False")

    best_pt = None
    best_clear = -1.0
    for _ in range(n_samples):
        p = pathfinder.get_random_navigable_point()
        if not np.all(np.isfinite(p)):
            continue
        c = float(pathfinder.distance_to_closest_obstacle(p))
        if c > best_clear:
            best_clear = c
            best_pt = np.asarray(p, dtype=np.float32).tolist()

    if best_pt is None or best_clear < min_clearance:
        raise RuntimeError(
            f"No navigable point with clearance >= {min_clearance} m "
            f"(best={best_clear:.2f} m over {n_samples} samples)"
        )
    return best_pt, best_clear


def equilateral_triangle_around(
    center: Sequence[float],
    radius: float,
    rotation: float = 0.0,
) -> List[Sequence[float]]:
    """Three points on the XZ circle of ``radius`` around ``center``,
    spaced 120 deg apart starting at angle ``rotation`` (radians, around +Y).
    Y of each point copies ``center[1]``."""
    pts: List[Sequence[float]] = []
    for k in range(3):
        theta = rotation + k * (2.0 * np.pi / 3.0)
        pts.append([
            float(center[0]) + radius * float(np.cos(theta)),
            float(center[1]),
            float(center[2]) + radius * float(np.sin(theta)),
        ])
    return pts
