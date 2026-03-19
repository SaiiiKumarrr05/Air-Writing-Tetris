"""
Optimized Gesture Recognition (Low Latency)
"""
from __future__ import annotations
import time
import numpy as np
from collections import deque


_HIST_LEN  = 14        # reduced history
_WINDOW_S  = 0.25      # faster response
_DEBOUNCE  = 0.25      # slightly faster re-trigger


class GestureRecognizer:
    def __init__(self, debounce: float = _DEBOUNCE):
        self._debounce = debounce
        self._wrist: deque[tuple[tuple, float]] = deque(maxlen=_HIST_LEN)
        self._index: deque[tuple[tuple, float]] = deque(maxlen=_HIST_LEN)
        self._fired: dict[str, float] = {}

    # --------------------------------------------------------------
    def update(self, wrist: tuple, index: tuple) -> None:
        t = time.time()
        self._wrist.append((wrist, t))
        self._index.append((index, t))

    def clear(self) -> None:
        self._wrist.clear()
        self._index.clear()

    # --------------------------------------------------------------
    def _ok(self, name: str) -> bool:
        now = time.time()
        if now - self._fired.get(name, 0.0) < self._debounce:
            return False
        self._fired[name] = now
        return True

    def _recent(self, hist):
        cutoff = time.time() - _WINDOW_S
        return [p for p, t in hist if t >= cutoff]

    # --------------------------------------------------------------
    def detect(self) -> str | None:
        pts = self._recent(self._wrist)
        if len(pts) < 4:
            return None

        x0, y0 = pts[0]
        x1, y1 = pts[-1]

        dx = x1 - x0
        dy = y1 - y0

        if abs(dx) > abs(dy) * 1.5:
            if dx > 45 and self._ok("right"):
                return "move_right"
            if dx < -45 and self._ok("left"):
                return "move_left"

        if dy > 55 and abs(dy) > abs(dx) * 1.5:
            if self._ok("drop"):
                return "fast_drop"

        return None

    # --------------------------------------------------------------
    def detect_rotate(self) -> bool:
        pts = self._recent(self._index)
        if len(pts) < 8:
            return False

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        w = max(xs) - min(xs)
        h = max(ys) - min(ys)

        # tighter bounds = less false positives + faster
        if w > 40 and h > 40 and 0.5 < (w / max(h, 1)) < 2.0:
            return self._ok("rotate")

        return False