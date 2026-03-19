"""
Optimized MediaPipe Hand Tracker (Lower Latency)
"""
from __future__ import annotations
import numpy as np
import cv2
import mediapipe as mp


class HandTracker:
    _LM = mp.solutions.hands.HandLandmark

    def __init__(self, min_detection: float = 0.65, min_tracking: float = 0.45):
        self._mp_hands = mp.solutions.hands
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,          # huge speed gain
            model_complexity=0,       # faster model
            min_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )

        _draw = mp.solutions.drawing_utils
        self._draw = _draw
        self._lm_spec  = _draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=2)
        self._con_spec = _draw.DrawingSpec(color=(80, 100, 255), thickness=1)

    # --------------------------------------------------------------
    def process(self, frame_bgr):
        """
        Downscale frame before processing to reduce latency.
        """
        small = cv2.resize(frame_bgr, None, fx=0.6, fy=0.6)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        return results, small.shape

    # --------------------------------------------------------------
    def _px(self, lm, lid: int, shape: tuple, scale_back=True):
        h, w = shape[:2]
        pt = lm.landmark[lid]

        x = int(pt.x * w)
        y = int(pt.y * h)

        if scale_back:
            x = int(x / 0.6)
            y = int(y / 0.6)

        return x, y

    def index_tip(self, lm, shape): return self._px(lm, self._LM.INDEX_FINGER_TIP, shape)
    def thumb_tip(self, lm, shape): return self._px(lm, self._LM.THUMB_TIP, shape)
    def wrist(self, lm, shape): return self._px(lm, self._LM.WRIST, shape)

    # --------------------------------------------------------------
    def is_pinching(self, lm, shape, thr: float = 38.0):
        ix, iy = self.index_tip(lm, shape)
        tx, ty = self.thumb_tip(lm, shape)
        return np.hypot(ix - tx, iy - ty) < thr

    # --------------------------------------------------------------
    def draw(self, frame, lm):
        self._draw.draw_landmarks(
            frame,
            lm,
            self._mp_hands.HAND_CONNECTIONS,
            self._lm_spec,
            self._con_spec,
        )

    def close(self):
        self.hands.close()