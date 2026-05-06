"""Server-side framework for VR virtual displays.

Goal: let mumt_hitl_app push *arbitrary* 2D content onto world-space or
HUD-attached quads inside the Unity Quest client without baking any
domain-specific logic into the APK. The client is a dumb canvas: it
gets create/update/setVisible/destroy events with image buffers (JPEG
bytes, base64-encoded) and just paints them.

Architecture::

    DisplayManager  -- registry, dispatcher, message-bus glue
        Display     -- abstract: knows how to produce a PIL.Image each tick
            SpotPovDisplay     -- first-person RGB camera bolted onto a Spot
            StaticTextDisplay  -- renders a callable's string output as JPEG
            ...                -- add new providers freely; no APK rebuild

Wire format (server -> client) lives under the ``mumtDisplays`` key in
the per-user keyframe message::

    {"mumtDisplays": {
        "create":     [{"uid": ..., "anchor": "head", "offset": [...],
                        "rotEuler": [...], "size": [...]}],
        "update":     [{"uid": ..., "imageBase64": "...", "format": "jpeg"}],
        "setVisible": [{"uid": ..., "visible": false}],
        "destroy":    ["..."]
    }}

Anchors are interpreted client-side:

    "head"        -- world-space quad parented to the XR camera (HUD-style)
    "world"       -- quad anchored at offset in habitat world coordinates
    "left_hand"   -- quad parented to the left controller
    "right_hand"  -- quad parented to the right controller

Bandwidth budget: 256x192 JPEG q70 ~= 8 kB/frame. Two POV displays at
15 fps ~= 240 kB/s, fine over adb-reverse localhost.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from base64 import b64encode
from dataclasses import dataclass, field
from io import BytesIO
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import habitat_sim
import magnum as mn

from habitat_hitl.core.user_mask import Mask

# cv2 is used by the optional TopDownMapDisplay; keep the import lazy so
# importers that only need DisplayManager + TextDisplay don't pay for it.
try:
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:  # pragma: no cover - cv2 is in our default deps but be safe
    _HAVE_CV2 = False


# Default resolution -- intentionally small. The Quest HUD is fine at this
# resolution and JPEGs stay under 10 kB. Override per-display via size_hw.
_DEFAULT_RES_HW: Tuple[int, int] = (192, 256)  # (height, width)
_DEFAULT_JPEG_QUALITY: int = 70


@dataclass
class DisplayLayout:
    """Where a display's quad lives in the client.

    All coordinates are in *Habitat world frame* (Y up, meters). The Unity
    client converts to its own LH coords before placing the quad.
    """
    anchor: str = "head"  # "head" | "world" | "left_hand" | "right_hand"
    offset: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    rot_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float] = (0.4, 0.3)  # (width_m, height_m)


class Display(ABC):
    """Abstract: owns one display, knows how to produce a PIL.Image each tick.

    Subclasses implement ``render()``. The base class handles fps throttling
    so providers don't all hammer the JPEG encoder at simulator rate.
    """

    def __init__(self, uid: str, layout: DisplayLayout, fps: float = 15.0):
        self.uid = uid
        self.layout = layout
        self.fps = float(fps)
        self._last_emit_sec: float = 0.0

    def _due(self, now_sec: float) -> bool:
        if self._last_emit_sec == 0.0:
            return True
        return (now_sec - self._last_emit_sec) >= (1.0 / max(self.fps, 0.1))

    @abstractmethod
    def render(self) -> Optional[Image.Image]:
        """Produce a PIL.Image (any mode; converted to RGB before encode).

        Return None to skip this update without changing the displayed
        frame on the client.
        """


class DisplayManager:
    """Server-side display registry. Pushes deltas through the HITL bus.

    Use one instance per AppState. Call ``tick()`` once per ``sim_update``;
    it renders any due frames and packs creates/updates/setVisible/destroy
    into the keyframe message that's about to be sent to clients.
    """

    def __init__(self, client_message_manager) -> None:
        # Held loosely so unit tests can pass None.
        self._cmm = client_message_manager
        self._displays: Dict[str, Display] = {}
        self._client_knows: set = set()
        self._visibility: Dict[str, bool] = {}
        self._pending_visibility: List[Tuple[str, bool]] = []
        self._pending_destroys: List[str] = []
        self._jpeg_quality: int = _DEFAULT_JPEG_QUALITY

    def add(self, display: Display) -> None:
        if display.uid in self._displays:
            raise ValueError(f"Display '{display.uid}' already registered")
        self._displays[display.uid] = display
        self._visibility[display.uid] = True

    def has(self, uid: str) -> bool:
        """Return True if a display with this uid is currently registered."""
        return uid in self._displays

    def set_visible(self, uid: str, visible: bool) -> None:
        if uid not in self._displays:
            return
        if self._visibility.get(uid, True) != visible:
            self._visibility[uid] = visible
            self._pending_visibility.append((uid, visible))

    def destroy(self, uid: str) -> None:
        if uid not in self._displays:
            return
        del self._displays[uid]
        self._visibility.pop(uid, None)
        self._pending_destroys.append(uid)

    def on_scene_change(self) -> None:
        """Reset client-knowledge when the client reloads. Re-emit creates."""
        self._client_knows.clear()

    # ------------------------------------------------------------------
    # Per-frame tick
    # ------------------------------------------------------------------

    def tick(self) -> None:
        if self._cmm is None:
            return
        now = monotonic()

        creates: List[Dict[str, Any]] = []
        updates: List[Dict[str, Any]] = []
        visibility_msgs: List[Dict[str, Any]] = []

        # 1) Emit create payloads for any not-yet-known displays.
        for uid, disp in self._displays.items():
            if uid in self._client_knows:
                continue
            creates.append({
                "uid": uid,
                "anchor": disp.layout.anchor,
                "offset": list(disp.layout.offset),
                "rotEuler": list(disp.layout.rot_euler_deg),
                "size": list(disp.layout.size),
            })
            self._client_knows.add(uid)
            print(f"[mumt-displays] queued create for '{uid}'", flush=True)

        # 2) Visibility deltas.
        for uid, vis in self._pending_visibility:
            visibility_msgs.append({"uid": uid, "visible": bool(vis)})
        self._pending_visibility.clear()

        # 3) Frame updates: only visible + due providers.
        for uid, disp in self._displays.items():
            if not self._visibility.get(uid, True):
                continue
            if not disp._due(now):
                continue
            try:
                img = disp.render()
            except Exception as exc:  # noqa: BLE001 - never let a provider kill the sim
                print(f"[mumt-displays] {uid}: render() raised {exc!r}", flush=True)
                continue
            if img is None:
                # Throttle the warn so a chronically-None provider doesn't
                # spam the log; once per ~2s is enough to tell us "this
                # display never produced pixels".
                last = getattr(disp, "_last_none_warn_sec", -1)
                cur_sec = int(now)
                if cur_sec - last >= 2:
                    print(
                        f"[mumt-displays] {uid}: render() returned None",
                        flush=True,
                    )
                    disp._last_none_warn_sec = cur_sec
                continue
            buf = BytesIO()
            try:
                img.convert("RGB").save(buf, format="JPEG", quality=self._jpeg_quality)
            except Exception as exc:  # noqa: BLE001
                print(f"[mumt-displays] {uid}: JPEG encode failed {exc!r}", flush=True)
                continue
            updates.append({
                "uid": uid,
                "imageBase64": b64encode(buf.getvalue()).decode("ascii"),
                "format": "jpeg",
            })
            disp._last_emit_sec = now

        # 4) Destroys: forget client-knowledge so future re-adds work.
        destroys = list(self._pending_destroys)
        self._pending_destroys.clear()
        for uid in destroys:
            self._client_knows.discard(uid)

        if not (creates or updates or visibility_msgs or destroys):
            return

        users = getattr(self._cmm, "_users", None)
        if users is None:
            return
        user_indices = list(users.indices(Mask.ALL))
        if creates:
            print(
                f"[mumt-displays] dispatching {len(creates)} create(s) "
                f"+{len(updates)} update(s) to users={user_indices}",
                flush=True,
            )
        # Periodic update-only diag so we can verify per-frame texture
        # delivery without spamming. Reports counts + total payload bytes
        # roughly once per second.
        if updates:
            self._tx_update_count = (
                getattr(self, "_tx_update_count", 0) + len(updates)
            )
            self._tx_update_bytes = (
                getattr(self, "_tx_update_bytes", 0)
                + sum(len(u.get("imageBase64", "")) for u in updates)
            )
            now_sec = int(now)
            if now_sec != getattr(self, "_tx_diag_last_sec", -1):
                print(
                    f"[mumt-displays] tx updates={self._tx_update_count} "
                    f"bytes_b64={self._tx_update_bytes} "
                    f"to users={user_indices}",
                    flush=True,
                )
                self._tx_diag_last_sec = now_sec
                self._tx_update_count = 0
                self._tx_update_bytes = 0
        for user_index in user_indices:
            msg = self._cmm.get_messages()[user_index]
            block = msg.setdefault("mumtDisplays", {})
            if creates:
                block.setdefault("create", []).extend(creates)
            if updates:
                block.setdefault("update", []).extend(updates)
            if visibility_msgs:
                block.setdefault("setVisible", []).extend(visibility_msgs)
            if destroys:
                block.setdefault("destroy", []).extend(destroys)


# ----------------------------------------------------------------------
# Concrete provider: first-person RGB camera bolted onto a Spot body.
# ----------------------------------------------------------------------

class SpotPovDisplay(Display):
    """First-person RGB camera attached to a Spot articulated object.

    We dynamically attach a habitat-sim CameraSensor to the spot AO's root
    scene node, with the camera offset from the body's local origin to
    approximate the Spot's "head" location. Each tick we call
    ``draw_observation`` + ``get_observation`` to read the rendered RGB
    into a PIL.Image.

    ``head_offset`` is in Spot's body-local frame: +X forward, +Y up,
    +Z right (matches the URDF convention used by ``mumt_sim.agents``).
    The defaults put the camera ~30 cm forward and ~50 cm above the body
    origin, roughly where Spot's stereo cameras live.

    The default ``orientation = (0, -pi/2, 0)`` rotates the camera so its
    forward (-Z in habitat sensor convention) aligns with the body's +X
    axis -- i.e. the camera looks where Spot is walking.
    """

    def __init__(
        self,
        uid: str,
        layout: DisplayLayout,
        *,
        sim: habitat_sim.Simulator,
        spot_ao,
        head_offset: Tuple[float, float, float] = (0.292, 0.50, 0.0),
        orientation_euler_rad: Tuple[float, float, float] = (
            0.0, -math.pi / 2.0, 0.0,
        ),
        size_hw: Tuple[int, int] = _DEFAULT_RES_HW,
        hfov_deg: float = 90.0,
        fps: float = 15.0,
        existing_sensor=None,
        draw_crosshair: bool = True,
        hud_text_fn: Optional[Callable[[], List[str]]] = None,
        hud_font_px: int = 18,
    ):
        super().__init__(uid, layout, fps=fps)
        self._sim = sim
        self._spot_ao = spot_ao
        self._draw_crosshair = bool(draw_crosshair)
        self._hud_text_fn = hud_text_fn
        self._hud_font: Optional[ImageFont.ImageFont] = None
        self._hud_font_px = int(hud_font_px)

        # Allow callers to share a sensor created elsewhere
        # (e.g. the autonomy stack's ``SpotHeadCam.color_sensor``) so
        # we don't pay for a redundant per-spot color sensor when both
        # the HUD and the coverage pipeline want the same raster.
        if existing_sensor is not None:
            self._sensor = existing_sensor
            return

        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = f"_mumt_display_{uid}"
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = list(size_hw)
        spec.hfov = float(hfov_deg)
        spec.position = list(head_offset)
        spec.orientation = list(orientation_euler_rad)
        self._sensor = sim.create_sensor(spec, spot_ao.root_scene_node)

    def render(self) -> Optional[Image.Image]:
        self._sensor.draw_observation()
        obs = self._sensor.get_observation()
        if obs is None:
            return None
        arr = np.asarray(obs)
        if not (arr.ndim == 3 and arr.shape[2] >= 3):
            return None
        img = Image.fromarray(arr[:, :, :3], "RGB")
        if self._draw_crosshair or self._hud_text_fn is not None:
            img = img.copy()  # PIL bound buffer is read-only
            draw = ImageDraw.Draw(img)
            if self._draw_crosshair:
                _draw_crosshair(draw, img.size)
            if self._hud_text_fn is not None:
                try:
                    lines = self._hud_text_fn() or []
                except Exception:  # noqa: BLE001 -- never let a HUD fn crash render
                    lines = []
                if lines:
                    if self._hud_font is None:
                        self._hud_font = _load_monospace_font(self._hud_font_px)
                    _draw_overlay_text(draw, img.size, lines, self._hud_font)
        return img


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    size_wh: Tuple[int, int],
    radius_px: int = 10,
    thickness_px: int = 2,
    color: Tuple[int, int, int] = (255, 255, 255),
    dot_radius_px: int = 2,
) -> None:
    """Draw a small "+" crosshair plus a center dot at image center.

    Bright white with a single-pixel black halo to stay readable
    against busy / bright scene backgrounds.
    """
    w, h = size_wh
    cx, cy = w // 2, h // 2
    halo = (0, 0, 0)

    # Crosshair arms with halo.
    for dx, dy, t in (
        (radius_px, 0, thickness_px),
        (0, radius_px, thickness_px),
    ):
        # Halo (slightly thicker, drawn first).
        draw.line(
            [(cx - dx, cy - dy), (cx + dx, cy + dy)],
            fill=halo, width=t + 2,
        )
    for dx, dy, t in (
        (radius_px, 0, thickness_px),
        (0, radius_px, thickness_px),
    ):
        draw.line(
            [(cx - dx, cy - dy), (cx + dx, cy + dy)],
            fill=color, width=t,
        )
    # Center dot.
    if dot_radius_px > 0:
        draw.ellipse(
            [
                (cx - dot_radius_px, cy - dot_radius_px),
                (cx + dot_radius_px, cy + dot_radius_px),
            ],
            fill=color, outline=halo,
        )


def _draw_overlay_text(
    draw: ImageDraw.ImageDraw,
    size_wh: Tuple[int, int],
    lines: Sequence[str],
    font: ImageFont.ImageFont,
    line_padding_px: int = 4,
    margin_px: int = 8,
) -> None:
    """Paint a translucent black strip at the top of the image and
    render ``lines`` in white inside it."""
    if not lines:
        return
    w, _ = size_wh
    line_h = max(font.size + line_padding_px, 14) if hasattr(font, "size") else 18
    n = len(lines)
    band_h = margin_px + n * line_h + margin_px
    # Translucent dark band (RGB approximate; PIL alpha would need RGBA).
    draw.rectangle(
        [(0, 0), (w, band_h)],
        fill=(0, 0, 0),
    )
    y = margin_px
    for line in lines:
        draw.text(
            (margin_px, y), str(line), fill=(255, 255, 255), font=font,
        )
        y += line_h


# ----------------------------------------------------------------------
# Concrete provider: alert wedge (solid color strip on FOV edge).
# ----------------------------------------------------------------------


class AlertWedgeDisplay(Display):
    """Solid-color strip anchored to the head's FOV edge.

    Used for "spot N raised an alert" attention grabs: the user's
    peripheral vision sees a bright red bar on the left (Spot 0) or
    right (Spot 1) edge of their HUD until acknowledged. Blinks to
    differentiate from static UI: the rendered pixel value alternates
    between full red and a darker red on a ~3 Hz cadence.

    Visibility is meant to be toggled by the host every tick (e.g.
    ``set_visible(False)`` after a 5 s timeout); this Display's job
    is just to produce the on-state image.
    """

    _RED_BRIGHT: Tuple[int, int, int] = (240, 30, 30)
    _RED_DIM: Tuple[int, int, int] = (130, 0, 0)

    def __init__(
        self,
        uid: str,
        layout: DisplayLayout,
        *,
        size_hw: Tuple[int, int] = (256, 32),
        blink_hz: float = 3.0,
        fps: float = 12.0,
    ):
        super().__init__(uid, layout, fps=fps)
        self._size_hw = (int(size_hw[0]), int(size_hw[1]))
        self._blink_hz = float(blink_hz)

    def render(self) -> Optional[Image.Image]:
        # Blink state derived from monotonic clock so all instances
        # blink in phase.
        phase = (monotonic() * self._blink_hz) % 1.0
        color = self._RED_BRIGHT if phase < 0.5 else self._RED_DIM
        h, w = self._size_hw
        return Image.new("RGB", (w, h), color=color)


# ----------------------------------------------------------------------
# Concrete provider: text panel (status, debug, etc.).
# ----------------------------------------------------------------------

_MONOSPACE_FONT_CANDIDATES: Tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)


def _load_monospace_font(size_px: int) -> ImageFont.ImageFont:
    """Return a TrueType monospace font at the requested pixel size, or
    fall back to PIL's tiny built-in bitmap font if no TTF is on disk."""
    for path in _MONOSPACE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size_px)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


class TextDisplay(Display):
    """Renders the output of ``text_fn()`` as a multiline text panel.

    Useful as a "smoke test" provider: cheap, no GPU, proves the whole
    pipeline works without depending on the simulator's render path. Also
    useful in production as a HUD readout (positions, mode names, etc.).
    """

    def __init__(
        self,
        uid: str,
        layout: DisplayLayout,
        text_fn: Callable[[], str],
        *,
        size_hw: Tuple[int, int] = (384, 512),
        bg: Tuple[int, int, int] = (12, 14, 22),
        fg: Tuple[int, int, int] = (220, 230, 220),
        font_px: int = 32,
        fps: float = 5.0,
    ):
        super().__init__(uid, layout, fps=fps)
        self._text_fn = text_fn
        self._size_hw = size_hw
        self._bg = bg
        self._fg = fg
        self._font = _load_monospace_font(font_px)

    def render(self) -> Optional[Image.Image]:
        h, w = self._size_hw
        img = Image.new("RGB", (w, h), self._bg)
        draw = ImageDraw.Draw(img)
        try:
            text = str(self._text_fn())
        except Exception as exc:  # noqa: BLE001
            text = f"<text_fn error: {exc}>"
        draw.multiline_text(
            (16, 16), text, fill=self._fg, font=self._font, spacing=6,
        )
        return img


# ----------------------------------------------------------------------
# Top-down map provider
# ----------------------------------------------------------------------

class TopDownMapDisplay(Display):
    """Live top-down map of the navmesh with agent dots + heading arrows.

    Constructs once: rasterises the navmesh into a binary occupancy grid
    and renders it to a BGR background image at the target output size.
    Each frame: copies the background and draws agent markers via cv2.
    Cheap (no GPU), works on any sim configuration including HITL's
    renderer-less SimDriver.

    ``pose_fn`` returns one tuple per agent::

        (world_x, world_z, yaw_rad, color_rgb)

    where ``color_rgb`` is a 0-255 (R, G, B) triple. Yaw convention
    matches ``mumt_sim.teleop``: yaw=0 means body forward along world +X,
    forward XZ vector = ``(cos yaw, -sin yaw)``.
    """

    def __init__(
        self,
        uid: str,
        layout: DisplayLayout,
        *,
        sim: habitat_sim.Simulator,
        pose_fn: Callable[
            [],
            Sequence[Tuple[float, float, float, Tuple[int, int, int]]],
        ],
        size_hw: Tuple[int, int] = (512, 512),
        cell_m: float = 0.10,
        nav_y_delta: float = 0.5,
        bg_nav_bgr: Tuple[int, int, int] = (90, 90, 90),
        bg_blocked_bgr: Tuple[int, int, int] = (24, 24, 28),
        fps: float = 5.0,
    ):
        if not _HAVE_CV2:
            raise RuntimeError(
                "TopDownMapDisplay requires opencv-python (cv2). Install with"
                " `pip install opencv-python` or use TextDisplay only.",
            )
        super().__init__(uid, layout, fps=fps)
        self._pose_fn = pose_fn

        b_min, b_max = sim.pathfinder.get_bounds()
        self._x_min, self._x_max = float(b_min[0]), float(b_max[0])
        self._z_min, self._z_max = float(b_min[2]), float(b_max[2])
        self._cell_m = float(cell_m)

        y_probe = self._discover_floor_y(sim)
        nx = max(1, int(math.ceil((self._x_max - self._x_min) / cell_m)))
        nz = max(1, int(math.ceil((self._z_max - self._z_min) / cell_m)))
        nav = np.zeros((nz, nx), dtype=bool)
        for j in range(nz):
            z = self._z_min + (j + 0.5) * cell_m
            for i in range(nx):
                x = self._x_min + (i + 0.5) * cell_m
                pt = np.array([x, y_probe, z], dtype=np.float32)
                nav[j, i] = bool(
                    sim.pathfinder.is_navigable(pt, max_y_delta=nav_y_delta)
                )

        # Pre-render the navmesh background (BGR) at the target size.
        bg = np.where(
            nav[..., None],
            np.array(bg_nav_bgr, dtype=np.uint8),
            np.array(bg_blocked_bgr, dtype=np.uint8),
        ).astype(np.uint8)

        # Aspect-correct fit: scale uniformly so the longer axis fills the
        # image, then center-pad the other axis. This avoids stretching a
        # rectangular scene into a square panel.
        target_h, target_w = size_hw
        sx = target_w / nx
        sz = target_h / nz
        s = min(sx, sz)
        out_w = max(1, int(round(nx * s)))
        out_h = max(1, int(round(nz * s)))
        scaled = cv2.resize(bg, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
        off_x = (target_w - out_w) // 2
        off_y = (target_h - out_h) // 2
        canvas[off_y:off_y + out_h, off_x:off_x + out_w] = scaled

        self._target_hw = (target_h, target_w)
        self._bg_bgr = canvas
        self._scale = s            # pixels per cell
        self._off_xy = (off_x, off_y)
        # Optional coverage-aware overlay. Patched in by the HITL app
        # via ``set_coverage_overlay`` once ``CoverageMap`` exists.
        # When wired, ``render()`` swaps the cached navmesh raster for
        # ``CoverageMap.render_topdown`` + ``draw_coarse_grid`` +
        # ``draw_spot_markers`` (matches ``agent_chat_multi_spot``).
        self._coverage_fn: Optional[Callable[[], Any]] = None
        self._sim_t_fn: Optional[Callable[[], float]] = None
        self._spot_colors_bgr: List[Tuple[int, int, int]] = [
            (0, 220, 255), (220, 80, 255),
        ]
        # Geometry of the coverage map's fine grid in pixel space; set
        # lazily on the first coverage-backed render so we can re-fit
        # if the coverage map's bounds differ from the navmesh probe.
        self._cov_geom: Optional[Dict[str, Any]] = None

    def set_coverage_overlay(
        self,
        coverage_fn: Callable[[], Any],
        sim_t_fn: Callable[[], float],
        spot_colors_bgr: Sequence[Tuple[int, int, int]],
    ) -> None:
        """Wire a CoverageMap-backed overlay after construction.

        ``coverage_fn()`` is called every render tick and may return
        ``None`` to fall back to the cached navmesh background.
        ``spot_colors_bgr`` should be one BGR tuple per Spot (n_spots
        entries), used both for the per-cell coverage tint and the
        body marker.
        """
        self._coverage_fn = coverage_fn
        self._sim_t_fn = sim_t_fn
        self._spot_colors_bgr = list(spot_colors_bgr)
        self._cov_geom = None  # force refit next render

    @staticmethod
    def _discover_floor_y(
        sim: habitat_sim.Simulator, n_samples: int = 64
    ) -> float:
        ys: List[float] = []
        for _ in range(n_samples):
            try:
                pt = sim.pathfinder.get_random_navigable_point()
            except Exception:  # noqa: BLE001
                continue
            if pt is None:
                continue
            y = float(pt[1])
            if math.isfinite(y):
                ys.append(y)
        if not ys:
            b_min, b_max = sim.pathfinder.get_bounds()
            return 0.5 * (float(b_min[1]) + float(b_max[1]))
        return float(np.median(ys))

    def _world_to_pix(self, wx: float, wz: float) -> Tuple[int, int]:
        gx = (wx - self._x_min) / self._cell_m
        gz = (wz - self._z_min) / self._cell_m
        cx = int(round(gx * self._scale)) + self._off_xy[0]
        cy = int(round(gz * self._scale)) + self._off_xy[1]
        return cx, cy

    def render(self) -> Optional[Image.Image]:
        try:
            poses = list(self._pose_fn())
        except Exception as exc:  # noqa: BLE001
            print(f"[mumt-displays] {self.uid}: pose_fn raised {exc!r}", flush=True)
            return None

        # ---------------- coverage-aware path ----------------
        # When a CoverageMap is wired in, use its built-in render +
        # sector grid + per-spot markers so the VR map matches the
        # look of agent_chat_multi_spot. We re-fit the coverage grid
        # into our target panel once, then re-render every tick.
        coverage = None
        if self._coverage_fn is not None:
            try:
                coverage = self._coverage_fn()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[mumt-displays] {self.uid}: coverage_fn raised "
                    f"{exc!r}", flush=True,
                )
                coverage = None
        sim_t = 0.0
        if self._sim_t_fn is not None:
            try:
                sim_t = float(self._sim_t_fn())
            except Exception:  # noqa: BLE001
                sim_t = 0.0

        if coverage is not None:
            target_h, target_w = self._target_hw
            # Fit the coverage fine grid into the target panel, padding
            # the shorter axis (matches render_coverage_pane in
            # agent_chat_multi_spot).
            if (
                self._cov_geom is None
                or self._cov_geom.get("nx") != coverage.nx
                or self._cov_geom.get("nz") != coverage.nz
            ):
                nx = int(coverage.nx)
                nz = int(coverage.nz)
                cov_scale = min(target_h / max(1, nz), target_w / max(1, nx))
                new_h = max(1, int(round(nz * cov_scale)))
                new_w = max(1, int(round(nx * cov_scale)))
                self._cov_geom = {
                    "nx": nx, "nz": nz,
                    "scale": float(cov_scale),
                    "new_h": new_h, "new_w": new_w,
                    "off_x": (target_w - new_w) // 2,
                    "off_y": (target_h - new_h) // 2,
                }

            geom = self._cov_geom
            # n_spots-many entries (truncate / pad with palette default).
            n = int(coverage.n_spots)
            colors = list(self._spot_colors_bgr[:n])
            while len(colors) < n:
                colors.append((255, 255, 255))

            try:
                fine = coverage.render_topdown(
                    t_now=sim_t, spot_colors_bgr=colors,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[mumt-displays] {self.uid}: render_topdown raised "
                    f"{exc!r}", flush=True,
                )
                return None

            upscaled = cv2.resize(
                fine,
                (geom["new_w"], geom["new_h"]),
                interpolation=cv2.INTER_NEAREST,
            )
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y0, x0 = geom["off_y"], geom["off_x"]
            canvas[y0:y0 + geom["new_h"], x0:x0 + geom["new_w"]] = upscaled
            view = canvas[
                y0:y0 + geom["new_h"], x0:x0 + geom["new_w"]
            ]
            try:
                coverage.draw_coarse_grid(view, cell_pixel_scale=geom["scale"])
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[mumt-displays] {self.uid}: draw_coarse_grid raised "
                    f"{exc!r}", flush=True,
                )

            # Build the spot-poses list from the wider ``poses`` callback
            # so the user (cyan) keeps showing too. CoverageMap.draw_spot_markers
            # only knows about the n_spots colour list, so we draw the
            # spots through it and the user as an extra circle.
            spot_poses = []
            for i in range(n):
                if i + 1 >= len(poses):
                    break
                # poses layout in mumt_hitl_app: [user, spot0, spot1, ...]
                wx, wz, yaw, _ = poses[i + 1]
                spot_poses.append((float(wx), float(wz), float(yaw)))
            if spot_poses:
                try:
                    coverage.draw_spot_markers(
                        view,
                        spot_poses=spot_poses,
                        spot_colors_bgr=colors[: len(spot_poses)],
                        cell_pixel_scale=geom["scale"],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[mumt-displays] {self.uid}: draw_spot_markers "
                        f"raised {exc!r}", flush=True,
                    )

            # Draw the user (poses[0]) as a slightly bigger ring so the
            # human pilot can find themselves at a glance.
            if poses:
                wx, wz, _yaw, color_rgb = poses[0]
                gx = (float(wx) - coverage.x_min) / coverage.cfg.fine_cell_m
                gz = (float(wz) - coverage.z_min) / coverage.cfg.fine_cell_m
                if 0 <= gx < coverage.nx and 0 <= gz < coverage.nz:
                    cx = int(round(gx * geom["scale"])) + x0
                    cy = int(round(gz * geom["scale"])) + y0
                    color_bgr = (
                        int(color_rgb[2]),
                        int(color_rgb[1]),
                        int(color_rgb[0]),
                    )
                    r = max(5, int(round(0.30 * geom["scale"] * 1.0)))
                    cv2.circle(canvas, (cx, cy), r + 2, (0, 0, 0),
                               thickness=2, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (cx, cy), r, color_bgr,
                               thickness=2, lineType=cv2.LINE_AA)

            out_rgb = canvas[..., ::-1].copy()
            return Image.fromarray(out_rgb, "RGB")

        # ---------------- legacy navmesh-only fallback ----------------
        out = self._bg_bgr.copy()
        radius_px = max(6, int(round(0.18 / self._cell_m * self._scale)))
        arrow_px = max(12, int(round(0.45 / self._cell_m * self._scale)))

        for entry in poses:
            wx, wz, yaw, color_rgb = entry
            cx, cy = self._world_to_pix(float(wx), float(wz))
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            tip_x = int(round(cx + arrow_px * math.cos(yaw)))
            tip_y = int(round(cy - arrow_px * math.sin(yaw)))
            cv2.circle(out, (cx, cy), radius_px + 2, (0, 0, 0),
                       thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(out, (cx, cy), radius_px, color_bgr,
                       thickness=-1, lineType=cv2.LINE_AA)
            cv2.arrowedLine(out, (cx, cy), (tip_x, tip_y), (0, 0, 0),
                            thickness=5, line_type=cv2.LINE_AA, tipLength=0.4)
            cv2.arrowedLine(out, (cx, cy), (tip_x, tip_y), color_bgr,
                            thickness=3, line_type=cv2.LINE_AA, tipLength=0.4)

        out_rgb = out[..., ::-1].copy()
        return Image.fromarray(out_rgb, "RGB")


# ----------------------------------------------------------------------
# Laser-pointer line: server-driven world-space line + endpoint sphere
# ----------------------------------------------------------------------
#
# Companion to MumtPointerLine.cs. Wire format:
#
#     {"mumtPointer": {"visible": bool,
#                      "originWorld": [x,y,z],
#                      "endpointWorld": [x,y,z],
#                      "colorRgb": [r,g,b]}}
#
# Coordinates are habitat world-space (Y up, Z back). The Unity client
# converts to its own coords via CoordinateSystem.ToUnityVector.
#
# Unlike DisplayManager, there's no create/destroy lifecycle: the line
# is a transient overlay that the server only emits while the user is
# actively pointing. On release we emit one final ``visible=false``
# packet and then go silent until the user points again.

class PointerKeyframe:
    """Server-side state for the VR laser pointer line.

    Use one instance per AppState. Each tick the host calls
    :meth:`set_visible` (with ``False`` to hide, or ``True`` along with
    ``origin`` / ``endpoint``) and then :meth:`tick` to flush the
    payload into every active client's keyframe message.

    The class only emits when state has actually changed (the rising
    edge of ``visible`` and every frame thereafter while ``visible`` is
    True), avoiding multi-kB/s of redundant messages while the trigger
    is idle.
    """

    def __init__(self, client_message_manager) -> None:
        self._cmm = client_message_manager
        self._visible: bool = False
        self._origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._endpoint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._color: Tuple[float, float, float] = (0.95, 0.20, 0.20)
        # Track whether we've already emitted a release packet; we want
        # exactly one ``visible=false`` per pointing session, then
        # silence, so the client's last-known-state semantics hold.
        self._was_visible: bool = False
        self._dirty: bool = False

    def set_visible(
        self,
        visible: bool,
        *,
        origin_world: Optional[Sequence[float]] = None,
        endpoint_world: Optional[Sequence[float]] = None,
        color_rgb: Optional[Sequence[float]] = None,
    ) -> None:
        """Update pointer state for the upcoming tick.

        ``origin_world`` / ``endpoint_world`` are habitat world-space
        triples (m). ``color_rgb`` is in [0, 1].
        """
        if visible:
            if origin_world is not None:
                self._origin = (float(origin_world[0]),
                                float(origin_world[1]),
                                float(origin_world[2]))
            if endpoint_world is not None:
                self._endpoint = (float(endpoint_world[0]),
                                  float(endpoint_world[1]),
                                  float(endpoint_world[2]))
            if color_rgb is not None:
                self._color = (float(color_rgb[0]),
                               float(color_rgb[1]),
                               float(color_rgb[2]))
            self._visible = True
            self._dirty = True
        else:
            self._visible = False
            # Only mark dirty if we owe the client a release packet.
            if self._was_visible:
                self._dirty = True

    def tick(self) -> None:
        """Flush the current state into the per-user keyframe message."""
        if self._cmm is None:
            return
        if not self._dirty:
            return
        users = getattr(self._cmm, "_users", None)
        if users is None:
            return
        user_indices = list(users.indices(Mask.ALL))
        payload: Dict[str, Any] = {"visible": bool(self._visible)}
        if self._visible:
            payload["originWorld"] = list(self._origin)
            payload["endpointWorld"] = list(self._endpoint)
            payload["colorRgb"] = list(self._color)
        for user_index in user_indices:
            msg = self._cmm.get_messages()[user_index]
            msg["mumtPointer"] = payload
        self._was_visible = self._visible
        # Edge-triggered: keep emitting while visible (endpoint moves),
        # but go silent after the single release packet has been sent.
        self._dirty = self._visible
