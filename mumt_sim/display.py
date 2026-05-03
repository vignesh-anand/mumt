"""OpenCV split-screen window for live habitat-sim teleop.

Window: ``cv2.imshow``. We deliberately avoid pygame because SDL's window
creates a GL context that conflicts with habitat-sim's GLX context on NVIDIA
drivers (segfault on the first ``sim.get_sensor_observations``). cv2's GTK/Qt
window has no GL state of its own, so habitat-sim's renderer keeps working.

Input: ``pynput.keyboard.Listener``. cv2's ``waitKey`` only fires at the OS's
keyboard auto-repeat rate, which on Linux begins after a ~250 ms delay. With
auto-repeat as the only signal, fresh keypresses had a noticeable dead-zone
between the initial press and when the OS started repeating. pynput hooks
real KeyPress/KeyRelease events at the X11 layer and gives us a true
held-key set with no auto-repeat dependency.

Caveats:
- pynput listens *globally*, so keys pressed while another window has focus
  still register here. Don't run this and edit code at the same time.
- pynput needs an X11 display (it uses Xlib via ``python-xlib``). Wayland
  without Xwayland will silently see no keys.

The window owns no policy: it returns a semantic ``InputState`` and the
teleop script translates it into ``TeleopInput``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np
from pynput import keyboard


# Canonical names the rest of the module uses to talk about keys.
# We normalise ``pynput`` events into this small string vocabulary so we don't
# leak ``pynput.Key.*`` symbols out of this file.
#
# Letter keys split into two groups:
# - "held" letters drive continuous teleop axes (W/S/A/D for forward/yaw,
#   plus R as edge for reset).
# - "primitive trigger" letters fire edge-only events for autonomous tools:
#   G = goto, M = move forward, N = turn (90 deg left), X = abort current
#   primitive on the active Spot.
_LETTER_KEYS = {"w", "s", "a", "d", "r", "g", "m", "n", "x"}
_NAV_KEYS = {"up", "down", "left", "right"}
_MOD_KEYS = {"shift"}

# Edge-triggered letter keys: rising edge produces a one-shot event,
# the held set still tracks them so we can debounce.
_EDGE_LETTERS = {"r", "g", "m", "n", "x"}


@dataclass
class InputState:
    """Per-frame snapshot of the user's keyboard intent."""

    forward: bool = False
    backward: bool = False
    yaw_left: bool = False
    yaw_right: bool = False
    pan_left: bool = False
    pan_right: bool = False
    tilt_up: bool = False
    tilt_down: bool = False
    boost: bool = False
    reset_pressed: bool = False          # edge-triggered (rising edge of R)
    switch_active_pressed: bool = False  # edge-triggered (rising edge of Tab)
    quit_pressed: bool = False           # Esc edge or window close
    # Primitive triggers (rising edge):
    goto_pressed: bool = False           # G
    move_pressed: bool = False           # M  (drive 1 m forward on active spot)
    turn_pressed: bool = False           # N  (turn 90 deg CCW on active spot)
    abort_pressed: bool = False          # X  (abort active spot's primitive)


class _PynputTracker:
    """Background thread that maintains a thread-safe held-key set.

    Receives real KeyPress / KeyRelease events from X11 via pynput, so 'is W
    held' is exactly that and not a stale-timestamp guess. Also tracks
    edge-triggered events for R (reset) and Esc (quit).
    """

    # Map "edge letter" name -> InputState field name. Keeps the
    # tracker generic over which letters fire edges.
    _EDGE_FIELDS = {
        "r": "reset",
        "g": "goto",
        "m": "move",
        "n": "turn",
        "x": "abort",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._edges: dict[str, bool] = {
            "reset": False,
            "switch": False,
            "quit": False,
            "goto": False,
            "move": False,
            "turn": False,
            "abort": False,
        }
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    @staticmethod
    def _normalize(key) -> Optional[str]:
        """Map a pynput key event onto our small canonical vocabulary."""
        if isinstance(key, keyboard.KeyCode):
            ch = key.char
            if ch:
                ch = ch.lower()
                if ch in _LETTER_KEYS:
                    return ch
            return None
        # SpecialKey enum
        if key == keyboard.Key.up:
            return "up"
        if key == keyboard.Key.down:
            return "down"
        if key == keyboard.Key.left:
            return "left"
        if key == keyboard.Key.right:
            return "right"
        if key in (keyboard.Key.shift, keyboard.Key.shift_r):
            return "shift"
        if key == keyboard.Key.tab:
            return "tab"
        if key == keyboard.Key.esc:
            return "esc"
        return None

    def _on_press(self, key) -> None:
        name = self._normalize(key)
        if name is None:
            return
        with self._lock:
            if name == "esc":
                self._edges["quit"] = True
                return
            if name == "tab" and "tab" not in self._held:
                self._edges["switch"] = True
            edge_field = self._EDGE_FIELDS.get(name)
            if edge_field is not None and name not in self._held:
                self._edges[edge_field] = True
            self._held.add(name)

    def _on_release(self, key) -> None:
        name = self._normalize(key)
        if name is None or name == "esc":
            return
        with self._lock:
            self._held.discard(name)

    def snapshot(self) -> tuple[set[str], dict[str, bool]]:
        """Return ``(held_keys_copy, edges)`` and clear the edges. The
        ``edges`` dict has one key per named edge:
        ``reset``, ``switch``, ``quit``, ``goto``, ``move``, ``turn``,
        ``abort``."""
        with self._lock:
            held = set(self._held)
            edges = dict(self._edges)
            for k in self._edges:
                self._edges[k] = False
        return held, edges

    def stop(self) -> None:
        try:
            self._listener.stop()
        except Exception:
            pass


class SplitScreenWindow:
    """Two-panel cv2 window + pynput-driven keyboard input.

    >>> win = SplitScreenWindow((480, 640), (480, 640), title="teleop")
    >>> while not win.should_close():
    ...     dt = win.tick(60)
    ...     state = win.poll_input()
    ...     win.show(left_rgb, right_rgb)
    """

    def __init__(
        self,
        left_hw,
        right_hw,
        title: str = "mumt",
    ) -> None:
        self.title = title
        self.left_h, self.left_w = int(left_hw[0]), int(left_hw[1])
        self.right_h, self.right_w = int(right_hw[0]), int(right_hw[1])

        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

        self._tracker = _PynputTracker()
        self._closed = False
        self._last_tick = time.monotonic()

    @staticmethod
    def _to_bgr(rgb: np.ndarray) -> np.ndarray:
        """habitat-sim hands us RGB (or RGBA) uint8; cv2 wants BGR."""
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    def _composite(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        hud_lines: Optional[Iterable[str]],
    ) -> np.ndarray:
        h = max(self.left_h, self.right_h)
        canvas = np.zeros((h, self.left_w + self.right_w, 3), dtype=np.uint8)
        canvas[: self.left_h, : self.left_w] = self._to_bgr(left_rgb)
        canvas[: self.right_h, self.left_w : self.left_w + self.right_w] = self._to_bgr(right_rgb)

        if hud_lines:
            y = 18
            for line in hud_lines:
                cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y += 18
        return canvas

    def show(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        hud_lines: Optional[Iterable[str]] = None,
    ) -> None:
        """Composite + display. Pair with ``tick`` so cv2 pumps its event loop."""
        canvas = self._composite(left_rgb, right_rgb, hud_lines)
        cv2.imshow(self.title, canvas)

    def tick(self, target_fps: int = 60) -> float:
        """Pace to ~``target_fps`` and return the elapsed dt in seconds.

        Pumps cv2's event loop via ``waitKey`` (cv2 needs this to refresh the
        window). Held-key state is tracked by pynput in a background thread,
        so any key cv2 happens to consume here is harmless.
        """
        target_dt = 1.0 / max(1, target_fps)
        now = time.monotonic()
        elapsed = now - self._last_tick
        sleep_ms = max(1, int(round((target_dt - elapsed) * 1000.0)))
        cv2.waitKey(sleep_ms)
        new_now = time.monotonic()
        dt = new_now - self._last_tick
        self._last_tick = new_now
        return dt

    def poll_input(self) -> InputState:
        """Return semantic axes from the current pynput-tracked held set."""
        try:
            visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0.0
        if visible < 1.0:
            self._closed = True

        held, edges = self._tracker.snapshot()
        if edges["quit"]:
            self._closed = True

        return InputState(
            forward="w" in held,
            backward="s" in held,
            yaw_left="a" in held,
            yaw_right="d" in held,
            pan_left="left" in held,
            pan_right="right" in held,
            tilt_up="up" in held,
            tilt_down="down" in held,
            boost="shift" in held,
            reset_pressed=edges["reset"],
            switch_active_pressed=edges["switch"],
            quit_pressed=self._closed,
            goto_pressed=edges["goto"],
            move_pressed=edges["move"],
            turn_pressed=edges["turn"],
            abort_pressed=edges["abort"],
        )

    def should_close(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._tracker.stop()
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass
        cv2.waitKey(1)


class MultiPaneWindow:
    """Grid-laid-out cv2 window + pynput-driven keyboard input.

    Generalisation of ``SplitScreenWindow`` for the autonomy harness. The
    constructor takes a row-major list of pane sizes; each row's height is
    the max of its panes' heights, and the canvas width is the max of any
    row's summed widths. Panes are indexed in row-major order (left-to-right,
    top-to-bottom) and ``show`` takes a flat list aligned with that order.

    Example: 3-pane layout with two head views on top and one wide coverage
    pane below:

    >>> win = MultiPaneWindow(pane_grid=[
    ...     [(360, 480), (360, 480)],   # top row: Spot 0 head | Spot 1 head
    ...     [(360, 960)],               # bottom row: coverage map
    ... ])
    >>> while not win.should_close():
    ...     dt = win.tick(60)
    ...     state = win.poll_input()
    ...     win.show([spot0, spot1, cov])
    """

    def __init__(
        self,
        pane_grid,
        title: str = "mumt",
    ) -> None:
        self.title = title
        self.pane_grid = [[(int(h), int(w)) for (h, w) in row] for row in pane_grid]
        if not self.pane_grid or not any(self.pane_grid):
            raise ValueError("pane_grid must contain at least one row with one pane")

        self._pane_placements: list[tuple[int, int, int, int]] = []  # (y, x, h, w)
        y = 0
        max_w = 0
        for row in self.pane_grid:
            if not row:
                raise ValueError("pane_grid rows must not be empty")
            row_max_h = max(h for (h, _) in row)
            x = 0
            for (h, w) in row:
                self._pane_placements.append((y, x, h, w))
                x += w
            max_w = max(max_w, x)
            y += row_max_h
        self._canvas_h = y
        self._canvas_w = max_w

        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

        self._tracker = _PynputTracker()
        self._closed = False
        self._last_tick = time.monotonic()

    @staticmethod
    def _to_bgr(img: np.ndarray) -> np.ndarray:
        """Accept RGB / RGBA / BGR uint8 and return BGR uint8 contiguous."""
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
            return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            # Heuristic: caller is responsible for telling us whether they
            # passed RGB or BGR. We treat 3-channel as RGB by default to match
            # SplitScreenWindow's contract (habitat-sim hands us RGB), and
            # downstream callers that already have BGR can just call
            # cv2.cvtColor(..., COLOR_BGR2RGB) before passing in - or we add a
            # convention if it gets noisy. For coverage map (which we draw in
            # BGR ourselves) we expose ``show_bgr`` below.
            return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        raise ValueError(f"unsupported image shape {img.shape}")

    @staticmethod
    def _draw_hud_at(canvas: np.ndarray, x: int, y0: int, lines) -> None:
        if not lines:
            return
        y = y0 + 18
        for line in lines:
            cv2.putText(canvas, line, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

    def show(
        self,
        panes,
        active_pane: Optional[int] = None,
        active_color_bgr=(0, 255, 0),
    ) -> None:
        """Composite + display.

        ``panes`` is a flat list aligned with the row-major pane order. Each
        entry is either:
          - a numpy image (RGB / RGBA / BGR-3ch / GRAY), or
          - a tuple ``(image, hud_lines, is_bgr=False)``.
        """
        if len(panes) != len(self._pane_placements):
            raise ValueError(
                f"expected {len(self._pane_placements)} panes, got {len(panes)}"
            )
        canvas = np.zeros((self._canvas_h, self._canvas_w, 3), dtype=np.uint8)
        for i, ((y, x, h, w), entry) in enumerate(zip(self._pane_placements, panes)):
            img = entry
            hud = None
            is_bgr = False
            if isinstance(entry, tuple):
                if len(entry) == 2:
                    img, hud = entry
                elif len(entry) == 3:
                    img, hud, is_bgr = entry
                else:
                    raise ValueError(f"pane {i}: tuple must be (img, hud[, is_bgr])")

            if is_bgr:
                if img.ndim == 2:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img_bgr = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]),
                                            cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = np.ascontiguousarray(img)
            else:
                img_bgr = self._to_bgr(img)

            ih, iw = img_bgr.shape[:2]
            if (ih, iw) != (h, w):
                img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
            canvas[y:y + h, x:x + w] = img_bgr

            if active_pane == i:
                cv2.rectangle(canvas, (x + 1, y + 1), (x + w - 2, y + h - 2),
                              active_color_bgr, 2)

            self._draw_hud_at(canvas, x, y, hud)

        cv2.imshow(self.title, canvas)

    def tick(self, target_fps: int = 60) -> float:
        target_dt = 1.0 / max(1, target_fps)
        now = time.monotonic()
        elapsed = now - self._last_tick
        sleep_ms = max(1, int(round((target_dt - elapsed) * 1000.0)))
        cv2.waitKey(sleep_ms)
        new_now = time.monotonic()
        dt = new_now - self._last_tick
        self._last_tick = new_now
        return dt

    def poll_input(self) -> InputState:
        try:
            visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0.0
        if visible < 1.0:
            self._closed = True

        held, edges = self._tracker.snapshot()
        if edges["quit"]:
            self._closed = True

        return InputState(
            forward="w" in held,
            backward="s" in held,
            yaw_left="a" in held,
            yaw_right="d" in held,
            pan_left="left" in held,
            pan_right="right" in held,
            tilt_up="up" in held,
            tilt_down="down" in held,
            boost="shift" in held,
            reset_pressed=edges["reset"],
            switch_active_pressed=edges["switch"],
            quit_pressed=self._closed,
            goto_pressed=edges["goto"],
            move_pressed=edges["move"],
            turn_pressed=edges["turn"],
            abort_pressed=edges["abort"],
        )

    def should_close(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._tracker.stop()
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass
        cv2.waitKey(1)


def make_placeholder_pane(hw, label: str) -> np.ndarray:
    """Build a dark grey BGR pane with centered text. Used while a real
    pane (e.g. the coverage map) hasn't been wired yet."""
    h, w = int(hw[0]), int(hw[1])
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx = max(0, (w - text_size[0]) // 2)
    ty = max(0, (h + text_size[1]) // 2)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 2, cv2.LINE_AA)
    return img
