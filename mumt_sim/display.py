"""Pygame-backed split-screen window for live habitat-sim teleop.

We deliberately keep this minimal:

- Two RGB panels (left + right) of arbitrary, possibly different ``(H, W)``.
- Per-frame ``show(left, right)`` blits both, ``tick(target_fps)`` paces the
  loop and returns the elapsed dt in seconds, ``poll_events()`` drains
  pygame's queue once per frame, and ``keys_pressed()`` exposes the held-key
  state for the teleop integrator.
- An optional HUD overlay (one short string) is drawn at the top-left.

Habitat-sim hands us ``(H, W, 4)`` uint8 arrays; pygame wants ``(W, H, 3)``,
so the conversion happens inside ``_to_surface``. Doing the slice / transpose
on a per-frame basis is fine at 480x640.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pygame


class SplitScreenWindow:
    """Two-panel pygame display with WASD-style input passthrough.

    >>> win = SplitScreenWindow((480, 640), (480, 640), title="teleop")
    >>> while True:
    ...     dt = win.tick(60)
    ...     events = win.poll_events()
    ...     win.show(left_rgb, right_rgb)
    """

    def __init__(
        self,
        left_hw: Sequence[int],
        right_hw: Sequence[int],
        title: str = "mumt",
        hud_font_size: int = 18,
    ) -> None:
        pygame.init()
        pygame.display.set_caption(title)

        self.left_h, self.left_w = int(left_hw[0]), int(left_hw[1])
        self.right_h, self.right_w = int(right_hw[0]), int(right_hw[1])
        screen_w = self.left_w + self.right_w
        screen_h = max(self.left_h, self.right_h)
        self._screen = pygame.display.set_mode((screen_w, screen_h))
        self._clock = pygame.time.Clock()
        self._hud_font = pygame.font.SysFont(None, hud_font_size)

    @staticmethod
    def _to_surface(rgb: np.ndarray) -> pygame.Surface:
        """Convert an ``(H, W, 3 or 4)`` uint8 array into a pygame Surface."""
        arr = rgb
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        # pygame.surfarray expects (W, H, 3); habitat gives (H, W, 3).
        return pygame.surfarray.make_surface(np.ascontiguousarray(arr.swapaxes(0, 1)))

    def show(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        hud_lines: Optional[Iterable[str]] = None,
    ) -> None:
        """Blit both frames and optional HUD lines, then flip."""
        self._screen.fill((0, 0, 0))
        self._screen.blit(self._to_surface(left_rgb), (0, 0))
        self._screen.blit(self._to_surface(right_rgb), (self.left_w, 0))

        if hud_lines:
            y = 4
            for line in hud_lines:
                surf = self._hud_font.render(line, True, (255, 255, 255), (0, 0, 0))
                self._screen.blit(surf, (6, y))
                y += surf.get_height() + 2

        pygame.display.flip()

    def tick(self, target_fps: int = 60) -> float:
        """Sleep to maintain ``target_fps`` and return elapsed dt in seconds."""
        return self._clock.tick(target_fps) / 1000.0

    def poll_events(self) -> list:
        """Drain pygame's event queue. Caller can scan for QUIT / KEYDOWN."""
        return pygame.event.get()

    def keys_pressed(self):
        """Held-key snapshot, indexable by ``pygame.K_*`` constants."""
        return pygame.key.get_pressed()

    def close(self) -> None:
        pygame.quit()
