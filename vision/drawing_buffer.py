"""
Stroke capture buffer with smooth Apple-Freeform style correction.

Pipeline:
Raw Stroke
→ Smoothing
→ Normalize to square
→ Continuous rasterization
→ Mild cleanup
→ 64x64 output for CNN
"""

from __future__ import annotations
import numpy as np
import cv2
from collections import deque

IMG_SIZE  = 64
_MIN_MOVE = 5
_SMOOTH_W = 5


class DrawingBuffer:
    def __init__(self, max_pts: int = 512, smooth_w: int = _SMOOTH_W):
        self._max_pts  = max_pts
        self._smooth_w = smooth_w
        self._pts: deque[tuple[float, float]] = deque(maxlen=max_pts)
        self._strokes: list[list[tuple[float, float]]] = []
        self._cur_stroke: list[tuple[float, float]] = []
        self._last: tuple[float, float] | None = None
        self.is_drawing = False

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def start_stroke(self) -> None:
        self.is_drawing  = True
        self._cur_stroke = []

    def end_stroke(self) -> None:
        self.is_drawing = False
        if len(self._cur_stroke) >= 5:
            self._strokes.append(list(self._cur_stroke))
        self._cur_stroke = []

    def add_point(self, pt: tuple[float, float]) -> None:
        if not self.is_drawing:
            return

        if self._last is not None:
            if np.hypot(pt[0] - self._last[0], pt[1] - self._last[1]) < _MIN_MOVE:
                return

        self._pts.append(pt)
        self._cur_stroke.append(pt)
        self._last = pt

    def clear(self) -> None:
        self._pts.clear()
        self._strokes.clear()
        self._cur_stroke.clear()
        self._last = None
        self.is_drawing = False

    def has_enough(self, min_pts: int = 20) -> bool:
        return len(self._pts) >= min_pts

    # ------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------
    def smoothed(self) -> list[tuple[float, float]]:
        pts = list(self._pts)
        if len(pts) < 4:
            return pts

        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)

        w = min(self._smooth_w, max(len(pts)//3, 1))
        if w > 1:
            k  = np.ones(w)/w
            xs = np.convolve(xs, k, mode="valid")
            ys = np.convolve(ys, k, mode="valid")

        return list(zip(xs.tolist(), ys.tolist()))

    # ------------------------------------------------------------
    # CNN-Compatible Rasterization
    # ------------------------------------------------------------
    def to_image(self, size: int = IMG_SIZE, thickness: int = 4) -> np.ndarray:
        """
        Returns 64x64 grayscale image:
        black background (0)
        white strokes (255)
        Matches synthetic training distribution.
        """

        pts = self.smoothed()
        canvas = np.zeros((size, size), dtype=np.uint8)

        if len(pts) < 2:
            return canvas

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        mnx, mxx = min(xs), max(xs)
        mny, mxy = min(ys), max(ys)

        # Preserve aspect ratio (critical)
        r = max(mxx - mnx, mxy - mny, 1e-6)
        pad = 8

        norm = [
            (
                int(pad + (x - mnx) / r * (size - 2 * pad)),
                int(pad + (y - mny) / r * (size - 2 * pad))
            )
            for x, y in pts
        ]

        # Draw smooth continuous stroke
        for i in range(1, len(norm)):
            cv2.line(canvas, norm[i-1], norm[i], 255, thickness)

        # Mild blur to match synthetic dataset
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        return canvas

    # ------------------------------------------------------------
    # Overlay Visualisation
    # ------------------------------------------------------------
    def draw_on_frame(self,
                      frame: np.ndarray,
                      offset_x: int = 0,
                      color: tuple = (0, 220, 180)) -> None:
        pts = self.smoothed()
        if len(pts) < 2:
            return

        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]) + offset_x, int(pts[i-1][1]))
            p2 = (int(pts[i][0])   + offset_x, int(pts[i][1]))
            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)

    # ------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------
    def is_partial(self) -> bool:
        return len(self._pts) < 15

    def aspect_ratio(self) -> float:
        pts = list(self._pts)
        if len(pts) < 4:
            return 1.0

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        return (max(xs)-min(xs)) / max(max(ys)-min(ys), 1)