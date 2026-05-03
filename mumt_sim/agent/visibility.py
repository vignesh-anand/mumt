"""Geometry helpers for the search-sector primitive.

Standalone numpy module so the controller in :mod:`tools` stays focused
on the state machine. Three things live here:

- :func:`los_visibility_mask` -- 360-degree shadow cast from a single
  fine-grid position against an ``is_navigable`` mask. Returns the set
  of cells reachable by an unblocked ray within ``max_range_cells``.
- :func:`fov_cone_mask` -- vectorised angular-cone test in fine-grid
  space, for picking out the slice of cells in front of the camera.
- :func:`visible_cells` -- composition of the two above intersected
  with a target mask.
- :func:`plan_search_tour` -- the actual search planner: random sample
  positions from a pose-space mask, evaluate visibility for 12 fixed
  headings each, then greedy set-cover up to K viewpoints.

Conventions (kept consistent with the rest of the codebase):

- ``is_navigable[iz, ix]`` -- first axis is the z grid index (rows),
  second is x (columns). Same as :class:`mumt_sim.agent.coverage.CoverageMap`.
- Yaw follows the body-frame convention used by ``SpotTeleop``: at
  yaw=0 the body's forward axis points along world +X. Per-cell heading
  to a target is ``atan2(-dz, dx)``, matching ``tools._heading_to``.
- Ray angle ``theta`` in shadow casting is the same body-frame angle:
  the per-step world XZ offset is ``(cos theta, -sin theta)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shadow-cast LOS
# ---------------------------------------------------------------------------


def los_visibility_mask(
    pos_idx: tuple[int, int],
    is_navigable: np.ndarray,
    max_range_cells: int,
    n_rays: int = 360,
) -> np.ndarray:
    """360-degree shadow cast from fine-grid cell ``pos_idx`` against
    ``is_navigable``.

    ``pos_idx`` is ``(iz, ix)`` (row, col) on the fine grid.
    ``is_navigable`` is the wall-proxy bool grid; non-navigable cells
    block rays.

    For each of ``n_rays`` directions we march one unit cell at a time
    out to ``max_range_cells``: any cell the ray *enters* before
    hitting a non-navigable cell is marked visible. The origin cell is
    always visible.

    Returns a bool ``(nz, nx)`` mask. Cost is O(n_rays * max_range)
    fully-vectorised numpy ops, ~180k ops for the default
    (n_rays=360, max_range_cells=50).
    """
    nz, nx = is_navigable.shape
    iz0, ix0 = int(pos_idx[0]), int(pos_idx[1])
    vis = np.zeros((nz, nx), dtype=bool)
    if not (0 <= iz0 < nz and 0 <= ix0 < nx):
        return vis
    vis[iz0, ix0] = True

    thetas = np.linspace(
        0.0, 2.0 * math.pi, num=int(n_rays), endpoint=False, dtype=np.float64
    )
    # Body-yaw -> per-step grid offsets: dix = cos(theta), diz = -sin(theta).
    dix = np.cos(thetas)
    diz = -np.sin(thetas)

    alive = np.ones(int(n_rays), dtype=bool)
    for s in range(1, int(max_range_cells) + 1):
        ix = np.rint(ix0 + s * dix).astype(np.int32)
        iz = np.rint(iz0 + s * diz).astype(np.int32)
        in_bounds = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
        # Kill rays that left the grid; clamp indices for the lookup.
        alive &= in_bounds
        if not alive.any():
            break
        ix_c = np.clip(ix, 0, nx - 1)
        iz_c = np.clip(iz, 0, nz - 1)
        cell_nav = is_navigable[iz_c, ix_c]
        # A ray is still alive at this step if it was alive AND landed
        # on a navigable cell. Mark visited cells before killing rays
        # that just hit a wall, so the wall cell itself stays unseen.
        step_alive = alive & cell_nav
        if step_alive.any():
            vis[iz_c[step_alive], ix_c[step_alive]] = True
        alive = step_alive
    return vis


# ---------------------------------------------------------------------------
# FOV cone
# ---------------------------------------------------------------------------


def fov_cone_mask(
    pos_idx: tuple[int, int],
    yaw_rad: float,
    hfov_rad: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Bool mask of fine cells whose angle-from-``pos_idx`` is within
    ``+/-hfov_rad/2`` of ``yaw_rad`` (body-frame).

    Range and LOS are NOT checked here -- intersect with
    :func:`los_visibility_mask` for that. The origin cell is included.
    """
    nz, nx = int(shape[0]), int(shape[1])
    iz0, ix0 = int(pos_idx[0]), int(pos_idx[1])
    iz_grid, ix_grid = np.meshgrid(
        np.arange(nz, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        indexing="ij",
    )
    dix = ix_grid - ix0
    diz = iz_grid - iz0
    # Reverse of the per-step convention used in shadow casting:
    # if a unit step at theta produces (dix, diz) = (cos, -sin), then
    # the angle to a cell at offset (dix, diz) is atan2(-diz, dix).
    angle = np.arctan2(-diz, dix)
    delta = np.mod(angle - float(yaw_rad) + math.pi, 2.0 * math.pi) - math.pi
    half = float(hfov_rad) * 0.5
    mask = np.abs(delta) <= half
    mask[iz0, ix0] = True
    return mask


# ---------------------------------------------------------------------------
# Composite visibility
# ---------------------------------------------------------------------------


def visible_cells(
    pos_idx: tuple[int, int],
    yaw_rad: float,
    target_mask: np.ndarray,
    is_navigable: np.ndarray,
    hfov_rad: float,
    max_range_cells: int,
    n_rays: int = 360,
    cached_los: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cells in ``target_mask`` that are visible from ``pos_idx`` at
    heading ``yaw_rad``: range-bounded, LOS-clear, inside the FOV
    cone.

    Pass ``cached_los`` to reuse a single LOS computation across many
    headings at the same position (the planner does this).
    """
    los = cached_los if cached_los is not None else los_visibility_mask(
        pos_idx, is_navigable, max_range_cells, n_rays=n_rays
    )
    cone = fov_cone_mask(pos_idx, yaw_rad, hfov_rad, target_mask.shape)
    return target_mask & los & cone


# ---------------------------------------------------------------------------
# Search planner
# ---------------------------------------------------------------------------


@dataclass
class Viewpoint:
    """One stop on the search tour."""

    x: float
    z: float
    yaw_rad: float
    expected_visible_cells: int


def plan_search_tour(
    coverage,
    sector_label: str,
    *,
    n_positions: int = 100,
    n_headings: int = 12,
    k_max: int = 6,
    min_gain_cells: int = 10,
    hfov_rad: float = math.radians(70.0),
    max_range_m: float = 5.0,
    n_los_rays: int = 360,
    rng: Optional[np.random.Generator] = None,
) -> list[Viewpoint]:
    """Random-sample greedy set-cover viewpoint planner.

    Workflow (mirrors the search-sector spec):

    1. Pose-space mask = navigable cells in the 3x3 block around
       ``sector_label`` (target sector + 8 neighbours).
    2. Target-cell set = navigable cells inside ``sector_label``.
       Treated as fully unseen.
    3. Sample ``n_positions`` positions uniformly from the pose-space
       mask. For each, compute one LOS visibility mask (360-degree
       shadow cast, independent of heading).
    4. For each (position, heading_k) with ``n_headings`` evenly-spaced
       headings, score = number of remaining target cells visible.
    5. Greedy: pick the (pos, yaw) with the largest score, append to
       the tour, remove its cells from "remaining", repeat up to
       ``k_max`` times. Stop early if the next-best gain falls below
       ``min_gain_cells``.

    Returns up to ``k_max`` :class:`Viewpoint`s in tour order. The
    caller is responsible for choosing the visit order (typically
    nearest-neighbour from the Spot's current pose).
    """
    rng = rng if rng is not None else np.random.default_rng()

    # Step 1: pose-space mask (3x3 block of sectors, navigable only).
    neighbour_lbls = coverage.neighbour_labels(sector_label, ring=1)
    pose_mask = coverage.region_navigable_mask(neighbour_lbls)

    # Step 2: target mask (navigable cells inside the target sector).
    ix_min, ix_max, iz_min, iz_max = coverage.sector_fine_indices(sector_label)
    target = np.zeros_like(coverage.is_navigable, dtype=bool)
    target[iz_min:iz_max, ix_min:ix_max] = coverage.is_navigable[
        iz_min:iz_max, ix_min:ix_max
    ]
    n_target = int(target.sum())
    if n_target == 0:
        return []

    # Step 3: sample positions uniformly from pose_mask.
    pose_iz, pose_ix = np.nonzero(pose_mask)
    if pose_iz.size == 0:
        return []
    n_sample = min(int(n_positions), pose_iz.size)
    sample_idx = rng.choice(pose_iz.size, size=n_sample, replace=False)
    sample_iz = pose_iz[sample_idx]
    sample_ix = pose_ix[sample_idx]

    # Step 4: compute LOS visibility once per sampled position, then
    # score each (position, heading) candidate against the target mask.
    fine_m = float(coverage.cfg.fine_cell_m)
    max_range_cells = max(1, int(math.ceil(float(max_range_m) / fine_m)))
    headings = np.linspace(
        0.0, 2.0 * math.pi, num=int(n_headings), endpoint=False
    )

    # vis_per_cand: list of (pos_index, heading_index, target-cell mask).
    # Memory: 1200 candidates * (50x50 bool) ~ 3 MB; fine.
    candidate_masks: list[np.ndarray] = []
    candidate_meta: list[tuple[int, int, float]] = []  # (iz, ix, yaw)
    for k in range(n_sample):
        iz, ix = int(sample_iz[k]), int(sample_ix[k])
        los = los_visibility_mask(
            (iz, ix), coverage.is_navigable, max_range_cells, n_rays=n_los_rays
        )
        # Restrict LOS to the target mask up front; FOV is computed below.
        los_target = los & target
        if not los_target.any():
            continue
        for yaw in headings:
            cone = fov_cone_mask((iz, ix), float(yaw), hfov_rad, target.shape)
            mask = los_target & cone
            if not mask.any():
                continue
            candidate_masks.append(mask)
            candidate_meta.append((iz, ix, float(yaw)))

    if not candidate_masks:
        return []

    # Step 5: greedy set-cover.
    remaining = target.copy()
    tour: list[Viewpoint] = []
    available = list(range(len(candidate_masks)))
    for _ in range(int(k_max)):
        best_idx = -1
        best_gain = 0
        best_mask: Optional[np.ndarray] = None
        for ci in available:
            gained = int(np.count_nonzero(candidate_masks[ci] & remaining))
            if gained > best_gain:
                best_gain = gained
                best_idx = ci
                best_mask = candidate_masks[ci]
        if best_idx < 0 or best_gain < int(min_gain_cells):
            break
        iz, ix, yaw = candidate_meta[best_idx]
        x = float(coverage.x_min + (ix + 0.5) * fine_m)
        z = float(coverage.z_min + (iz + 0.5) * fine_m)
        tour.append(
            Viewpoint(
                x=x, z=z, yaw_rad=float(yaw),
                expected_visible_cells=int(best_gain),
            )
        )
        if best_mask is not None:
            remaining = remaining & ~best_mask
        available.remove(best_idx)
        if not remaining.any():
            break

    return tour
