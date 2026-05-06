"""Navmesh-aware spawn utilities."""
from __future__ import annotations

import math
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
    max_floor_dy: float = 0.30,
    top_k: int = 10,
    rng: "np.random.Generator | None" = None,
) -> Tuple[Sequence[float], float]:
    """Search the navmesh for an open *floor-level* navigable point.

    Returns ``(point, clearance)`` where ``clearance`` is the distance in
    metres from ``point`` to the nearest obstacle (wall or static collider).

    Samples ``n_samples`` random navigable points, filters them to within
    ``max_floor_dy`` metres of the median sample Y (which gives us the
    floor height in HSSD scenes -- sofas, beds, etc. show up as elevated
    minority navmesh patches), keeps every candidate with clearance
    >= ``min_clearance``, and picks **one at random from the top-``top_k``**
    by clearance.

    Picking from the top-K rather than the single max means restarting
    the server gets you a different spawn each time -- useful when the
    global maximum-clearance point happens to be inside furniture
    geometry that ``distance_to_closest_obstacle`` doesn't account for
    (HSSD has e.g. low couches whose seat is at floor Y, with a wide
    free hemisphere above that fools the obstacle metric).

    Pass ``rng`` to seed the tiebreaker; defaults to fresh randomness.

    Raises RuntimeError if no sample meets ``min_clearance``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not pathfinder.is_loaded:
        raise RuntimeError("pathfinder.is_loaded == False")

    # Pass 1: sample, record candidates with their clearances + ys.
    pts: list = []
    ys: list = []
    clears: list = []
    for _ in range(n_samples):
        p = pathfinder.get_random_navigable_point()
        if not np.all(np.isfinite(p)):
            continue
        c = float(pathfinder.distance_to_closest_obstacle(p))
        if not math.isfinite(c):
            continue
        pts.append(np.asarray(p, dtype=np.float32))
        ys.append(float(p[1]))
        clears.append(c)

    if not pts:
        raise RuntimeError(
            f"No navigable points sampled (out of {n_samples})"
        )

    floor_y = float(np.median(ys))

    qualified: list = []
    for p, y, c in zip(pts, ys, clears):
        if abs(y - floor_y) > max_floor_dy:
            continue
        if c >= min_clearance:
            qualified.append((c, p))

    if not qualified:
        best_clear = max(clears) if clears else -1.0
        raise RuntimeError(
            f"No floor-level (|dy|<={max_floor_dy} m vs median {floor_y:.2f}) "
            f"navigable point with clearance >= {min_clearance} m "
            f"(best={best_clear:.2f} m over {n_samples} samples)"
        )

    # Sort by clearance descending and pick uniformly from the top-K.
    qualified.sort(key=lambda cp: -cp[0])
    pool = qualified[: max(1, int(top_k))]
    idx = int(rng.integers(0, len(pool)))
    chosen_clear, chosen_pt = pool[idx]
    return chosen_pt.tolist(), float(chosen_clear)


def _geodesic_distance(pathfinder, a, b) -> float:
    """Shortest-path distance along the navmesh between ``a`` and ``b``.

    Returns ``+inf`` if no path exists (e.g. the points are in disconnected
    navmesh islands). Both endpoints must already be on the navmesh.
    """
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    if not pathfinder.find_path(path):
        return float("inf")
    return float(path.geodesic_distance)


def sample_far_pair_navmesh(
    pathfinder,
    n_samples: int = 600,
    min_clearance: float = 0.7,
    refine_iters: int = 3,
) -> Tuple[Sequence[float], Sequence[float], float]:
    """Find a pair of navigable points that are far apart *along the navmesh*.

    Picks an anchor as the highest-clearance point in a random sample, then
    iteratively re-anchors on the partner with the longest geodesic distance
    from the current anchor (a couple of iterations of the standard
    "double-sweep" diameter approximation on the navmesh graph). All
    candidates are filtered by ``min_clearance`` so neither spot ends up
    pressed against a wall.

    Returns ``(point_a, point_b, geodesic_distance_m)``.

    Raises RuntimeError if fewer than two candidates pass the clearance
    filter, or if no pair is path-connected on the navmesh.
    """
    if not pathfinder.is_loaded:
        raise RuntimeError("pathfinder.is_loaded == False")

    candidates: List[np.ndarray] = []
    for _ in range(n_samples):
        p = pathfinder.get_random_navigable_point()
        if not np.all(np.isfinite(p)):
            continue
        if float(pathfinder.distance_to_closest_obstacle(p)) < min_clearance:
            continue
        candidates.append(np.asarray(p, dtype=np.float32))

    if len(candidates) < 2:
        raise RuntimeError(
            f"Only {len(candidates)} navmesh samples passed clearance "
            f">= {min_clearance} m (out of {n_samples}); cannot pick a pair."
        )

    cands_arr = np.stack(candidates, axis=0)

    clearances = np.array(
        [float(pathfinder.distance_to_closest_obstacle(p)) for p in cands_arr]
    )
    anchor_idx = int(np.argmax(clearances))
    anchor = cands_arr[anchor_idx]

    partner = anchor
    best_d = 0.0
    for _ in range(max(1, refine_iters)):
        ds = np.array(
            [_geodesic_distance(pathfinder, anchor, c) for c in cands_arr]
        )
        ds[~np.isfinite(ds)] = -1.0
        partner_idx = int(np.argmax(ds))
        new_d = float(ds[partner_idx])
        if new_d <= best_d:
            break
        partner = cands_arr[partner_idx]
        best_d = new_d
        anchor, partner = partner, anchor

    if best_d <= 0.0:
        raise RuntimeError(
            "Could not find any pair of navigable points connected by a "
            "navmesh path. Scene navmesh may be entirely disconnected."
        )

    return anchor.tolist(), partner.tolist(), best_d


def navmesh_path_midpoint(
    pathfinder, a: Sequence[float], b: Sequence[float]
) -> Sequence[float]:
    """Point on the shortest navmesh path from ``a`` to ``b`` closest to the
    halfway mark by arc length. Returns a 3-vector snapped to the navmesh.

    Falls back to ``snap_point((a+b)/2)`` if the shortest path has no
    intermediate waypoints (e.g. very short paths).
    """
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    if not pathfinder.find_path(path) or len(path.points) < 2:
        mid = (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)) / 2.0
        return np.asarray(pathfinder.snap_point(mid)).tolist()

    pts = [np.asarray(p, dtype=np.float32) for p in path.points]
    seg_lens = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
    total = sum(seg_lens)
    if total <= 1e-6:
        return np.asarray(pts[0]).tolist()

    target = total / 2.0
    travelled = 0.0
    for i, L in enumerate(seg_lens):
        if travelled + L >= target:
            t = (target - travelled) / max(L, 1e-6)
            mid = pts[i] + t * (pts[i + 1] - pts[i])
            return np.asarray(pathfinder.snap_point(mid)).tolist()
        travelled += L
    return np.asarray(pts[-1]).tolist()


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
