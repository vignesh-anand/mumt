"""Top-down coverage map for the autonomy harness.

A two-tier grid spanning the navmesh AABB on the XZ plane:

- **Fine grid** (``fine_cell_m``, default 10 cm). Per-cell ``is_navigable`` is
  computed once at construction by sampling ``pathfinder.is_navigable`` at
  each cell centre. ``last_seen_t_per_spot`` is a float[nz, nx, n_spots]
  array updated each tick by back-projecting each Spot's depth pixels into
  world XZ and stamping the cell each pixel hits.
- **Coarse 1 m chess-named grid** -- this slice (B) lays the substrate but
  doesn't surface the coarse rollup yet. That comes in slice C; the
  fine-grid storage and update path defined here are the only thing the
  coarse layer ever reads.

Update model uses the depth sensor directly rather than ``pathfinder.cast_ray``
because (a) it's much cheaper (one numpy back-projection vs a Python loop of
ray casts) and (b) the depth sensor IS the ground-truth visibility for that
camera pose, so the result is exactly "cells whose floor / wall / object
geometry the camera actually sees", with occlusion handled for free.

Coordinate convention:
- Render image shape is ``(nz, nx, 3)``: row index = z cell, col index = x
  cell. World +X is right, world +Z is down. Matches the natural top-down
  "north up" feel when paired with habitat-sim's left-handed XYZ.
- Camera convention follows habitat-sim: camera looks down -Z in its local
  frame, +Y is up, +X is right. The depth sensor returns z-depth (distance
  along the camera's -Z axis), not Euclidean distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

import habitat_sim
import magnum as mn


@dataclass
class CoverageMapConfig:
    fine_cell_m: float = 0.10
    """Fine grid resolution in meters."""

    max_range_m: float = 5.0
    """Depth pixels beyond this are ignored (treated as out-of-range)."""

    nav_probe_y_delta: float = 1.5
    """Vertical tolerance passed to ``pathfinder.is_navigable``. Generous
    enough to absorb minor floor-height variation (steps, thresholds) and
    slight differences between our probe Y and the true floor."""

    nav_probe_y: Optional[float] = None
    """Vertical sampling height for the navigability check, in world Y. If
    ``None`` (default) we auto-discover the floor by sampling a handful of
    random navigable points from the pathfinder and taking their median Y
    -- this is robust on HSSD where the AABB midpoint Y often lies far
    above the floor (since the AABB includes the ceiling)."""

    pixel_stride: int = 4
    """Subsample factor for the depth image during back-projection. 1 = use
    every pixel; 4 = use every 4th pixel in both axes (i.e. 1/16 the work).
    At 480x640 a stride of 4 gives ~19k samples per Spot per tick which is
    well inside numpy's free budget."""

    coarse_cell_m: float = 5.0
    """Coarse-grid cell size in metres -- the agent's spatial sector
    vocabulary. Used by ``draw_coarse_grid`` to overlay chess-style
    grid lines and ``A1`` / ``B7`` labels on the rendered map, and by
    ``coarse_label_for_world_xz`` to resolve a world point to its
    sector. Independent of the fine cell size used for occupancy
    bookkeeping."""


def _matrix4_to_numpy(m: mn.Matrix4) -> np.ndarray:
    """Convert a magnum.Matrix4 to a numpy ``(4, 4)`` row-major float32.

    magnum stores matrices column-major and indexes by column:
    ``m[i]`` returns column ``i`` as a Vector4. We pull each column out and
    build the row-major numpy matrix manually to avoid any ambiguity.
    """
    out = np.empty((4, 4), dtype=np.float32)
    for col in range(4):
        v = m[col]
        out[0, col] = float(v[0])
        out[1, col] = float(v[1])
        out[2, col] = float(v[2])
        out[3, col] = float(v[3])
    return out


class CoverageMap:
    """Per-fine-cell ``last_seen_t`` map driven by depth back-projection.

    >>> cov = CoverageMap(sim, n_spots=2, config=CoverageMapConfig())
    >>> cam_T_world = cov.head_camera_world_transform(sim, agent_id=1, sensor_uuid="spot_0_head_depth")
    >>> cov.update_from_depth(spot_id=0, t_now=1.0, cam_T_world=cam_T_world,
    ...                       depth=depth_obs, hfov_deg=110.0)
    >>> cov.stamp_self_cell(spot_id=0, t_now=1.0, world_xyz=spot_pos)
    >>> img_bgr = cov.render_topdown(t_now=1.0,
    ...                              spot_colors_bgr=[(255, 255, 0), (255, 0, 255)])
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        n_spots: int,
        config: Optional[CoverageMapConfig] = None,
    ) -> None:
        self.cfg = config or CoverageMapConfig()
        self.n_spots = int(n_spots)

        b_min, b_max = sim.pathfinder.get_bounds()
        self.x_min = float(b_min[0])
        self.x_max = float(b_max[0])
        self.z_min = float(b_min[2])
        self.z_max = float(b_max[2])

        # Pick a probe Y that's actually on the floor. HSSD AABBs include
        # the ceiling so the midpoint is often 1.5+ m above the floor, which
        # makes is_navigable miss almost every cell.
        if self.cfg.nav_probe_y is not None:
            self.y_probe = float(self.cfg.nav_probe_y)
        else:
            self.y_probe = self._discover_floor_y(sim)

        c = self.cfg.fine_cell_m
        self.nx = max(1, int(math.ceil((self.x_max - self.x_min) / c)))
        self.nz = max(1, int(math.ceil((self.z_max - self.z_min) / c)))

        self.is_navigable = self._build_navigability(sim)
        self.last_seen_t = np.full(
            (self.nz, self.nx, self.n_spots), -1e9, dtype=np.float32
        )

        # Camera-space ray cache, keyed by (h, w, hfov_deg).
        self._ray_cache: dict[tuple[int, int, float], np.ndarray] = {}

    # ----- construction helpers -------------------------------------------

    @staticmethod
    def _discover_floor_y(
        sim: habitat_sim.Simulator, n_samples: int = 64
    ) -> float:
        """Sample random navigable points from the pathfinder and return the
        median Y. This is the robust way to get the floor height on HSSD,
        whose AABB midpoint is typically far above the floor."""
        ys = []
        for _ in range(n_samples):
            try:
                pt = sim.pathfinder.get_random_navigable_point()
            except Exception:
                continue
            if pt is None:
                continue
            y = float(pt[1])
            if math.isfinite(y):
                ys.append(y)
        if not ys:
            # No navmesh? Fall back to AABB midpoint and hope max_y_delta
            # covers the gap.
            b_min, b_max = sim.pathfinder.get_bounds()
            return 0.5 * (float(b_min[1]) + float(b_max[1]))
        return float(np.median(ys))

    def _build_navigability(self, sim: habitat_sim.Simulator) -> np.ndarray:
        """Sample pathfinder.is_navigable at each fine cell centre."""
        nav = np.zeros((self.nz, self.nx), dtype=bool)
        c = self.cfg.fine_cell_m
        max_dy = self.cfg.nav_probe_y_delta
        xs = self.x_min + (np.arange(self.nx, dtype=np.float32) + 0.5) * c
        zs = self.z_min + (np.arange(self.nz, dtype=np.float32) + 0.5) * c
        for j, z in enumerate(zs):
            for i, x in enumerate(xs):
                pt = np.array([x, self.y_probe, z], dtype=np.float32)
                nav[j, i] = bool(sim.pathfinder.is_navigable(pt, max_y_delta=max_dy))
        return nav

    # ----- coordinate helpers ---------------------------------------------

    def world_xz_to_cell(
        self, x: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised: ``(gx, gz, in_bounds_mask)`` for arrays of world X, Z."""
        gx = np.floor((x - self.x_min) / self.cfg.fine_cell_m).astype(np.int32)
        gz = np.floor((z - self.z_min) / self.cfg.fine_cell_m).astype(np.int32)
        ok = (gx >= 0) & (gx < self.nx) & (gz >= 0) & (gz < self.nz)
        return gx, gz, ok

    @staticmethod
    def head_camera_world_transform(
        sim: habitat_sim.Simulator, agent_id: int, sensor_uuid: str
    ) -> np.ndarray:
        """Return the camera's world transform as a numpy ``(4, 4)`` matrix.

        Walks the full scene-node chain (agent body * sensor local tilt) so
        the result reflects pan AND tilt for that frame. Pair with the
        camera-space rays from ``_get_camera_rays`` to back-project depth
        into the world.
        """
        agent = sim.get_agent(agent_id)
        sensor = agent._sensors[sensor_uuid]
        T_mn = sensor.node.absolute_transformation()
        return _matrix4_to_numpy(T_mn)

    def _get_camera_rays(self, h: int, w: int, hfov_deg: float) -> np.ndarray:
        """``(h, w, 3)`` camera-space coordinates of the back-projected point
        at unit depth, i.e. ``(x_cam/d, y_cam/d, -1)`` for each pixel.

        Multiplying by the depth at that pixel yields the camera-space
        point. Square pixels (``fy == fx``) are assumed; fine for our
        habitat-sim cameras.
        """
        key = (int(h), int(w), float(hfov_deg))
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached
        hfov = math.radians(hfov_deg)
        fx = (w / 2.0) / math.tan(hfov / 2.0)
        fy = fx  # square pixels
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)  # (h, w)
        # +x_cam: right; +y_cam: up (image v=0 is top, so flip); -z_cam: forward
        x_over_d = (uu - cx) / fx
        y_over_d = -(vv - cy) / fy
        z_over_d = -np.ones_like(uu)
        rays = np.stack([x_over_d, y_over_d, z_over_d], axis=-1).astype(np.float32)
        self._ray_cache[key] = rays
        return rays

    # ----- update path ----------------------------------------------------

    def update_from_depth(
        self,
        spot_id: int,
        t_now: float,
        cam_T_world: np.ndarray,
        depth: np.ndarray,
        hfov_deg: float,
        body_xz: Optional[tuple[float, float]] = None,
    ) -> int:
        """Stamp every navigable cell that this Spot's depth sensor sees.

        If ``body_xz`` is supplied we additionally require an unblocked
        straight-line path on the navmesh from the body's XZ to each
        candidate cell -- this filters out the "head pokes through a thin
        wall" artifact (the head camera offset is 0.479 m forward of the
        body, larger than the 0.3 m navmesh agent radius, so the head can
        end up on the far side of a thin wall while the body is still
        flush against it). Cells that are visible from the camera but
        unreachable in a straight body-frame line are dropped.

        Returns the number of unique cells stamped this call (handy for
        debug HUDs).
        """
        if depth.ndim != 2:
            raise ValueError(f"expected (H, W) depth, got {depth.shape}")
        h, w = depth.shape
        rays = self._get_camera_rays(h, w, hfov_deg)
        stride = max(1, int(self.cfg.pixel_stride))
        d = depth[::stride, ::stride]
        r = rays[::stride, ::stride]

        valid = (d > 1e-3) & (d < self.cfg.max_range_m) & np.isfinite(d)
        if not valid.any():
            return 0
        d_v = d[valid].astype(np.float32).reshape(-1, 1)        # (N, 1)
        r_v = r[valid].reshape(-1, 3)                           # (N, 3)
        # Camera-space points: (x_cam/d, y_cam/d, -1) * d = (x_cam, y_cam, -d)
        cam_pts = r_v * d_v                                     # (N, 3)
        N = cam_pts.shape[0]
        cam_pts_h = np.empty((N, 4), dtype=np.float32)
        cam_pts_h[:, :3] = cam_pts
        cam_pts_h[:, 3] = 1.0
        # World point = cam_T_world @ p (with p as a column vector). Equivalent
        # to (p_h_row @ cam_T_world.T)[:, :3].
        world_pts = (cam_pts_h @ cam_T_world.T)[:, :3]          # (N, 3)
        x = world_pts[:, 0]
        z = world_pts[:, 2]
        gx, gz, ok = self.world_xz_to_cell(x, z)
        gx = gx[ok]
        gz = gz[ok]
        if gx.size == 0:
            return 0
        nav_mask = self.is_navigable[gz, gx]
        gx = gx[nav_mask]
        gz = gz[nav_mask]
        if gx.size == 0:
            return 0

        # Deduplicate: many depth pixels map to the same cell.
        flat = gz.astype(np.int64) * self.nx + gx.astype(np.int64)
        unique_flat = np.unique(flat)
        u_gz = (unique_flat // self.nx).astype(np.int32)
        u_gx = (unique_flat % self.nx).astype(np.int32)

        if body_xz is not None:
            body_gx = int(np.clip(
                math.floor((float(body_xz[0]) - self.x_min) / self.cfg.fine_cell_m),
                0, self.nx - 1,
            ))
            body_gz = int(np.clip(
                math.floor((float(body_xz[1]) - self.z_min) / self.cfg.fine_cell_m),
                0, self.nz - 1,
            ))
            los = self._line_of_sight_mask(body_gx, body_gz, u_gx, u_gz)
            u_gx = u_gx[los]
            u_gz = u_gz[los]
            if u_gx.size == 0:
                return 0

        # Plain assignment is fine: t_now is identical for every entry, no
        # race between rows.
        self.last_seen_t[u_gz, u_gx, spot_id] = t_now
        return int(u_gx.size)

    def _line_of_sight_mask(
        self,
        body_gx: int,
        body_gz: int,
        target_gx: np.ndarray,
        target_gz: np.ndarray,
    ) -> np.ndarray:
        """For each ``(target_gx[i], target_gz[i])`` candidate cell, return
        True if the straight body->target line walked at fine-cell resolution
        passes only through navigable cells.

        Vectorised over all targets: builds an ``(N, L)`` array of cell
        coordinates sampled along each line (where L is the longest line in
        cells), looks up ``is_navigable[zs, xs]``, and AND-reduces along the
        path axis.
        """
        N = int(target_gx.shape[0])
        if N == 0:
            return np.zeros(0, dtype=bool)
        dx = target_gx.astype(np.int32) - body_gx
        dz = target_gz.astype(np.int32) - body_gz
        max_len = int(max(np.max(np.abs(dx)), np.max(np.abs(dz)))) + 1
        if max_len < 2:
            return np.ones(N, dtype=bool)
        t = np.linspace(0.0, 1.0, max_len, dtype=np.float32).reshape(1, max_len)
        xs = (body_gx + dx.reshape(N, 1).astype(np.float32) * t).astype(np.int32)
        zs = (body_gz + dz.reshape(N, 1).astype(np.float32) * t).astype(np.int32)
        # Defensive: clamp to bounds.
        np.clip(xs, 0, self.nx - 1, out=xs)
        np.clip(zs, 0, self.nz - 1, out=zs)
        nav = self.is_navigable[zs, xs]   # (N, max_len) bool
        return nav.all(axis=1)

    def stamp_self_cell(
        self, spot_id: int, t_now: float, world_xyz: Sequence[float]
    ) -> bool:
        """Force-stamp the cell the Spot is standing in.

        Useful so the cell directly under the Spot is always coloured even
        when the head is angled steeply away from it. Returns True if the
        cell was in-bounds and navigable.
        """
        x = np.asarray([float(world_xyz[0])], dtype=np.float32)
        z = np.asarray([float(world_xyz[2])], dtype=np.float32)
        gx, gz, ok = self.world_xz_to_cell(x, z)
        if not bool(ok[0]):
            return False
        j, i = int(gz[0]), int(gx[0])
        if not bool(self.is_navigable[j, i]):
            return False
        self.last_seen_t[j, i, spot_id] = t_now
        return True

    # ----- rendering ------------------------------------------------------

    def render_topdown(
        self,
        t_now: float,
        spot_colors_bgr: Sequence[tuple[int, int, int]],
        fresh_age_s: float = 30.0,
        max_age_s: float = 300.0,
        min_alpha: float = 0.30,
        background_navigable_bgr: tuple[int, int, int] = (48, 48, 48),
        background_blocked_bgr: tuple[int, int, int] = (16, 16, 16),
    ) -> np.ndarray:
        """Return a ``(nz, nx, 3)`` BGR uint8 image of the coverage state.

        Per cell the most recent ``last_seen_t`` across spots picks the
        dominant colour; per-spot tints are alpha-blended on top of the
        navigable / blocked background. Alpha is full-bright for ``age <=
        fresh_age_s``, decays linearly to ``min_alpha`` by ``max_age_s``.
        """
        if len(spot_colors_bgr) < self.n_spots:
            raise ValueError(
                f"need {self.n_spots} colours, got {len(spot_colors_bgr)}"
            )

        nav = self.is_navigable
        bg_nav = np.array(background_navigable_bgr, dtype=np.uint8)
        bg_blocked = np.array(background_blocked_bgr, dtype=np.uint8)
        img = np.where(nav[..., None], bg_nav, bg_blocked).astype(np.uint8)

        denom = max(1e-6, max_age_s - fresh_age_s)
        for spot_id in range(self.n_spots):
            color = np.array(spot_colors_bgr[spot_id], dtype=np.float32)
            t_seen = self.last_seen_t[..., spot_id]
            seen = t_seen > -1e8
            if not seen.any():
                continue
            age = np.where(seen, t_now - t_seen, np.inf)
            alpha = 1.0 - ((age - fresh_age_s) / denom) * (1.0 - min_alpha)
            alpha = np.clip(alpha, min_alpha, 1.0)
            alpha = np.where(seen, alpha, 0.0).astype(np.float32)
            base = img.astype(np.float32)
            blended = base * (1.0 - alpha[..., None]) + color * alpha[..., None]
            img = blended.astype(np.uint8)

        return img

    def draw_spot_markers(
        self,
        img: np.ndarray,
        spot_poses: Sequence[tuple[float, float, float]],
        spot_colors_bgr: Sequence[tuple[int, int, int]],
        marker_radius_cells: float = 4.0,
        arrow_length_cells: float = 10.0,
        cell_pixel_scale: float = 1.0,
    ) -> np.ndarray:
        """Draw a body-position circle + heading arrow per Spot, in-place.

        ``spot_poses`` is a list of ``(world_x, world_z, yaw_rad)`` tuples
        (one per Spot). Pass ``cell_pixel_scale`` = (image pixels per
        fine cell) when drawing on an upscaled render; markers and the
        arrow stroke are sized at that scale so they stay smooth and
        proportional.

        Yaw convention matches mumt_sim.teleop: yaw=0 means body forward
        along world +X, and the world-XZ forward vector is
        ``(cos yaw, -sin yaw)``. In the image we map +X to columns (right)
        and +Z to rows (down), so image-space forward = ``(cos, -sin)``.
        """
        if len(spot_poses) != len(spot_colors_bgr):
            raise ValueError("spot_poses and spot_colors_bgr length mismatch")
        s = float(cell_pixel_scale)
        radius_px = max(2, int(round(marker_radius_cells * s)))
        arrow_px = float(arrow_length_cells * s)
        outline_thick = max(2, int(round(0.6 * s)) + 2)
        stroke_thick = max(1, outline_thick - 1)
        out = img  # we draw in-place; cv2 funcs require contiguous BGR
        for (wx, wz, yaw), color in zip(spot_poses, spot_colors_bgr):
            gx = (float(wx) - self.x_min) / self.cfg.fine_cell_m
            gz = (float(wz) - self.z_min) / self.cfg.fine_cell_m
            if not (0 <= gx < self.nx and 0 <= gz < self.nz):
                continue
            cx = int(round(gx * s))
            cy = int(round(gz * s))
            tip_x = int(round(cx + arrow_px * math.cos(yaw)))
            tip_y = int(round(cy - arrow_px * math.sin(yaw)))
            color_t = tuple(int(c) for c in color)
            cv2.circle(out, (cx, cy), radius_px, (0, 0, 0),
                       thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(out, (cx, cy), max(1, radius_px - 1),
                       color_t, thickness=-1, lineType=cv2.LINE_AA)
            cv2.arrowedLine(out, (cx, cy), (tip_x, tip_y), (0, 0, 0),
                            thickness=outline_thick, line_type=cv2.LINE_AA,
                            tipLength=0.4)
            cv2.arrowedLine(out, (cx, cy), (tip_x, tip_y), color_t,
                            thickness=stroke_thick, line_type=cv2.LINE_AA,
                            tipLength=0.4)
        return out

    @staticmethod
    def _coarse_col_label(i: int) -> str:
        """0->A, 1->B, ..., 25->Z, 26->AA, 27->AB, ..."""
        n = i + 1
        label = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            label = chr(ord("A") + r) + label
        return label

    def coarse_label_for_world_xz(self, x: float, z: float) -> Optional[str]:
        """Chess-style label (e.g. ``"C7"``) for the coarse cell that
        contains world point ``(x, z)``, or ``None`` if out of bounds.

        Coarse-cell origin is the AABB ``(x_min, z_min)`` corner, so
        ``A1`` is the top-left coarse cell when looking at the rendered
        map (low X, low Z). Column letter follows X (left -> right);
        row number follows Z (top -> bottom)."""
        coarse_m = float(self.cfg.coarse_cell_m)
        col = int(math.floor((float(x) - self.x_min) / coarse_m))
        row = int(math.floor((float(z) - self.z_min) / coarse_m))
        n_cols = int(math.ceil(self.nx * self.cfg.fine_cell_m / coarse_m))
        n_rows = int(math.ceil(self.nz * self.cfg.fine_cell_m / coarse_m))
        if not (0 <= col < n_cols and 0 <= row < n_rows):
            return None
        return f"{self._coarse_col_label(col)}{row + 1}"

    @staticmethod
    def _parse_coarse_label(label: str) -> tuple[int, int]:
        """Reverse of ``_coarse_col_label`` + the row suffix. Returns
        ``(col, row)`` (zero-based). Raises ``ValueError`` on garbage."""
        s = str(label).strip().upper()
        if not s:
            raise ValueError("empty coarse label")
        i = 0
        while i < len(s) and s[i].isalpha():
            i += 1
        letters, digits = s[:i], s[i:]
        if not letters or not digits or not digits.isdigit():
            raise ValueError(
                f"coarse label {label!r} must look like 'C7' (letters + digits)"
            )
        col = 0
        for ch in letters:
            col = col * 26 + (ord(ch) - ord("A") + 1)
        col -= 1
        row = int(digits) - 1
        if col < 0 or row < 0:
            raise ValueError(f"coarse label {label!r} parses to negative index")
        return col, row

    def world_xz_for_coarse_label(self, label: str) -> tuple[float, float]:
        """World ``(x, z)`` of the centre of the coarse cell named by
        ``label`` (e.g. ``"C7"``). Inverse of
        :meth:`coarse_label_for_world_xz`. Raises ``ValueError`` if the
        label is malformed or refers to a cell outside the AABB."""
        col, row = self._parse_coarse_label(label)
        coarse_m = float(self.cfg.coarse_cell_m)
        n_cols = int(math.ceil(self.nx * self.cfg.fine_cell_m / coarse_m))
        n_rows = int(math.ceil(self.nz * self.cfg.fine_cell_m / coarse_m))
        if not (0 <= col < n_cols and 0 <= row < n_rows):
            raise ValueError(
                f"coarse label {label!r} -> ({col}, {row}) is outside the "
                f"AABB grid of {n_cols} cols x {n_rows} rows"
            )
        x = self.x_min + (col + 0.5) * coarse_m
        z = self.z_min + (row + 0.5) * coarse_m
        return float(x), float(z)

    def draw_coarse_grid(
        self,
        img: np.ndarray,
        cell_pixel_scale: float = 1.0,
        line_color_bgr: tuple[int, int, int] = (170, 170, 170),
        label_color_bgr: tuple[int, int, int] = (240, 240, 240),
        label_bg_bgr: Optional[tuple[int, int, int]] = (0, 0, 0),
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Overlay coarse ``coarse_cell_m`` grid lines + chess-style
        labels in-place on ``img``.

        Pass ``cell_pixel_scale`` = (image pixels per fine cell) when
        drawing on an upscaled render so the lines align with cells.
        Column letters (A, B, ...) march along +X (left -> right) and
        row numbers (1, 2, ...) march along +Z (top -> bottom)."""
        coarse_m = float(self.cfg.coarse_cell_m)
        fine_per_coarse = max(1, int(round(coarse_m / self.cfg.fine_cell_m)))
        s = float(cell_pixel_scale)
        coarse_px = fine_per_coarse * s

        h, w = img.shape[:2]
        n_cols = int(math.ceil(self.nx / fine_per_coarse))
        n_rows = int(math.ceil(self.nz / fine_per_coarse))
        max_x = int(round(min(w - 1, self.nx * s)))
        max_y = int(round(min(h - 1, self.nz * s)))

        for i in range(n_cols + 1):
            x = int(round(i * coarse_px))
            if 0 <= x <= max_x:
                cv2.line(img, (x, 0), (x, max_y), line_color_bgr,
                         1, cv2.LINE_AA)
        for j in range(n_rows + 1):
            y = int(round(j * coarse_px))
            if 0 <= y <= max_y:
                cv2.line(img, (0, y), (max_x, y), line_color_bgr,
                         1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(n_cols):
            cx = int(round((i + 0.5) * coarse_px))
            if not (0 <= cx <= max_x):
                continue
            label = self._coarse_col_label(i)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            tx = cx - tw // 2
            ty = th + 3
            if label_bg_bgr is not None:
                cv2.rectangle(
                    img,
                    (tx - 2, ty - th - 2),
                    (tx + tw + 2, ty + 2),
                    label_bg_bgr,
                    thickness=-1,
                )
            cv2.putText(img, label, (tx, ty), font, font_scale,
                        label_color_bgr, 1, cv2.LINE_AA)
        for j in range(n_rows):
            cy = int(round((j + 0.5) * coarse_px))
            if not (0 <= cy <= max_y):
                continue
            label = str(j + 1)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            tx = 3
            ty = cy + th // 2
            if label_bg_bgr is not None:
                cv2.rectangle(
                    img,
                    (tx - 2, ty - th - 2),
                    (tx + tw + 2, ty + 2),
                    label_bg_bgr,
                    thickness=-1,
                )
            cv2.putText(img, label, (tx, ty), font, font_scale,
                        label_color_bgr, 1, cv2.LINE_AA)

        return img
