"""OpenCV split-screen window for live habitat-sim teleop.

We deliberately avoid pygame here: SDL's window creates its own GL context
which conflicts with habitat-sim's GLX context on NVIDIA drivers, producing
a segfault during ``sim.get_sensor_observations``. ``cv2.imshow`` uses a
GTK/Qt window with no GL state of its own, so habitat-sim's renderer keeps
working.

Trade-off: cv2 only reports the most recent key per ``waitKey`` call (no native
'is-key-held' query). We emulate held-key state by stamping each keycode's
most-recent press time and treating it as held for a short ``hold_window`` of
~120 ms. Linux key auto-repeat fires at ~30 Hz once started, which keeps the
stamp fresh as long as the key is physically down. The first ~250 ms after
press (before auto-repeat kicks in) does feel sticky; that is OS-level keyboard
delay, not something we can hide.

The window owns *no* policy: it returns an ``InputState`` with semantic axis
flags, and the teleop script translates that into ``TeleopInput``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


# X11 keysym codes returned by ``cv2.waitKeyEx`` for arrow keys on Linux.
# cv2 returns the X keysym OR'd with 0xff0000 on some versions; we mask down.
_KEY_LEFT = 0xff51
_KEY_UP = 0xff52
_KEY_RIGHT = 0xff53
_KEY_DOWN = 0xff54

# Letter keys map to their ASCII codes (lower- and uppercase distinct).
_KEY_W, _KEY_w = ord("W"), ord("w")
_KEY_S, _KEY_s = ord("S"), ord("s")
_KEY_A, _KEY_a = ord("A"), ord("a")
_KEY_D, _KEY_d = ord("D"), ord("d")
_KEY_R, _KEY_r = ord("R"), ord("r")
_KEY_ESC = 27

_FORWARD_KEYS = {_KEY_w, _KEY_W}
_BACKWARD_KEYS = {_KEY_s, _KEY_S}
_YAW_LEFT_KEYS = {_KEY_a, _KEY_A}
_YAW_RIGHT_KEYS = {_KEY_d, _KEY_D}
_BOOST_KEYS = {_KEY_W, _KEY_S, _KEY_A, _KEY_D}  # uppercase = shift held
_RESET_KEYS = {_KEY_r, _KEY_R}


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
    reset_pressed: bool = False  # edge-triggered
    quit_pressed: bool = False   # Esc or window close


class SplitScreenWindow:
    """Two-panel cv2 window with held-key emulation.

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
        hold_window_s: float = 0.12,
    ) -> None:
        self.title = title
        self.left_h, self.left_w = int(left_hw[0]), int(left_hw[1])
        self.right_h, self.right_w = int(right_hw[0]), int(right_hw[1])
        self._hold_window = float(hold_window_s)

        # WINDOW_AUTOSIZE keeps the panel pixels 1:1 (no resampling jitter).
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

        # Map keycode -> last-seen monotonic time. We never evict; the dict
        # caps at the keys the user has touched, ~tens of entries.
        self._last_seen: dict[int, float] = {}
        self._reset_was_held = False
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
                # Black outline + white fill for readability over any panel.
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
        """Composite + display. Must be followed by a poll_input/tick to pump cv2."""
        canvas = self._composite(left_rgb, right_rgb, hud_lines)
        cv2.imshow(self.title, canvas)

    def tick(self, target_fps: int = 60) -> float:
        """Sleep to maintain ~``target_fps`` and return elapsed dt in seconds.

        cv2 doesn't have a ``Clock`` so we hand-roll: compute remaining ms to
        the next frame deadline, then call ``waitKey`` to both pace and pump
        events. ``waitKey``'s return value is consumed inside ``poll_input``.
        """
        target_dt = 1.0 / max(1, target_fps)
        now = time.monotonic()
        elapsed = now - self._last_tick
        sleep_ms = max(1, int(round((target_dt - elapsed) * 1000.0)))
        # ``waitKey`` also pumps the GUI so the window stays responsive. We
        # don't consume the keycode here; ``poll_input`` calls ``waitKey(1)``
        # again to drain whatever is fresh. cv2 only buffers the most recent
        # keypress so this is fine in practice.
        cv2.waitKey(sleep_ms)
        new_now = time.monotonic()
        dt = new_now - self._last_tick
        self._last_tick = new_now
        return dt

    def _drain_keys(self) -> list[int]:
        """Read every keycode cv2 has buffered for us this tick."""
        codes: list[int] = []
        for _ in range(8):  # bounded just in case
            k = cv2.waitKeyEx(1)
            if k == -1:
                break
            codes.append(k)
        return codes

    def _is_held(self, key: int, now: float) -> bool:
        ts = self._last_seen.get(key)
        return ts is not None and (now - ts) < self._hold_window

    def _any_held(self, keys, now: float) -> bool:
        return any(self._is_held(k, now) for k in keys)

    def poll_input(self) -> InputState:
        """Drain pending keys, update held-state, return semantic axes."""
        # Window-close detection. cv2.WND_PROP_VISIBLE goes to 0 when the user
        # clicks the X. (On some builds it stays 1 forever - users can also
        # press Esc as the canonical quit.)
        try:
            visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0.0
        if visible < 1.0:
            self._closed = True

        now = time.monotonic()
        for code in self._drain_keys():
            if code == _KEY_ESC:
                self._closed = True
            self._last_seen[code] = now

        state = InputState(
            forward=self._any_held(_FORWARD_KEYS, now),
            backward=self._any_held(_BACKWARD_KEYS, now),
            yaw_left=self._any_held(_YAW_LEFT_KEYS, now),
            yaw_right=self._any_held(_YAW_RIGHT_KEYS, now),
            pan_left=self._is_held(_KEY_LEFT, now),
            pan_right=self._is_held(_KEY_RIGHT, now),
            tilt_up=self._is_held(_KEY_UP, now),
            tilt_down=self._is_held(_KEY_DOWN, now),
            boost=self._any_held(_BOOST_KEYS, now),
            quit_pressed=self._closed,
        )

        # Edge-trigger reset: rising edge of any R variant.
        reset_now = self._any_held(_RESET_KEYS, now)
        state.reset_pressed = reset_now and not self._reset_was_held
        self._reset_was_held = reset_now

        return state

    def should_close(self) -> bool:
        return self._closed

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass
        cv2.waitKey(1)
