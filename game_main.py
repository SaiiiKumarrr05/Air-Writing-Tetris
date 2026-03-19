"""
Air Writing Tetris — AR Overlay Edition (Engine-Aligned Final Version)
"""

from __future__ import annotations
import cv2
import numpy as np

from auto_train import ensure_model
ensure_model()

from game.engine import TetrisEngine
from vision.hand_tracker import HandTracker
from vision.drawing_buffer import DrawingBuffer
from vision.gesture_recognizer import GestureRecognizer
from models.classifier import TetrominoClassifier
from utils.renderer import FONT, WHITE, YELLOW

DRAWING_ZONE_FRAC = 0.65
ASK_LABEL_CONF    = 0.45
PIECE_NAMES       = ["I", "O", "T", "S", "Z", "J", "L"]

# Colors (BGR)
NEON_GREEN  = (0, 255, 120)
NEON_BLUE   = (255, 180, 0)
CYAN        = (255, 255, 0)


class AirTetris:
    def __init__(self):
        self.engine     = TetrisEngine()
        self.tracker    = HandTracker()
        self.buf        = DrawingBuffer()
        self.gesture    = GestureRecognizer()
        self.clf        = TetrominoClassifier()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.state        = "drawing"
        self.top_pred     = []
        self.drawing_img  = None
        self.ask_label    = False
        self.pending      = None
        self.pending_conf = 0.0

        self.cell_size = 25
        self.board_w = 10 * self.cell_size
        self.board_h = 20 * self.cell_size

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    def run(self) -> None:
        cv2.namedWindow("Air Tetris AR", cv2.WINDOW_NORMAL)

        while True:
            ret, cam = self.cap.read()
            if not ret:
                break

            cam = cv2.flip(cam, 1)
            h, w = cam.shape[:2]

            self._process_vision(cam)
            self._tick_engine()

            overlay = cam.copy()

            board_x = (w - self.board_w) // 2
            board_y = (h - self.board_h) // 2

            self._draw_grid(overlay, board_x, board_y)
            self._draw_pieces(overlay, board_x, board_y)
            self._draw_hud(overlay, w, h)

            out = cv2.addWeighted(cam, 0.6, overlay, 0.4, 0)

            self._draw_ui_text(out, w, h, board_x)

            if self.ask_label:
                out = self._draw_label_picker(out)

            cv2.imshow("Air Tetris AR", out)

            if not self._handle_key(cv2.waitKey(1) & 0xFF):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # ==========================================================
    # DRAWING
    # ==========================================================
    def _draw_grid(self, canvas, x, y):
        cv2.rectangle(canvas, (x, y), (x + self.board_w, y + self.board_h), (10, 10, 10), -1)

        for i in range(11):
            cv2.line(canvas,
                     (x + i*self.cell_size, y),
                     (x + i*self.cell_size, y + self.board_h),
                     (50, 50, 50), 1)

        for i in range(21):
            cv2.line(canvas,
                     (x, y + i*self.cell_size),
                     (x + self.board_w, y + i*self.cell_size),
                     (50, 50, 50), 1)

    def _draw_pieces(self, canvas, ox, oy):
        # Draw locked blocks
        for r, row in enumerate(self.engine.grid):
            for c, val in enumerate(row):
                if val is not None:
                    self._draw_block(
                        canvas,
                        ox + c*self.cell_size,
                        oy + r*self.cell_size,
                        val
                    )

        # Draw active falling piece (FIXED HERE)
        if self.engine.current:
            for r, c in self.engine.current.absolute_cells():
                if r >= 0:
                    self._draw_block(
                        canvas,
                        ox + c*self.cell_size,
                        oy + r*self.cell_size,
                        self.engine.current.name
                    )

    def _draw_block(self, canvas, x, y, name):
        color = self._get_piece_color(name)
        cv2.rectangle(canvas,
                      (x+1, y+1),
                      (x+self.cell_size-1, y+self.cell_size-1),
                      color, -1)
        cv2.rectangle(canvas,
                      (x+1, y+1),
                      (x+self.cell_size-1, y+self.cell_size-1),
                      WHITE, 1)

    def _draw_hud(self, canvas, w, h):
        dz_w = int(w * DRAWING_ZONE_FRAC)
        dz_x = (w - dz_w) // 2
        dz_color = NEON_GREEN if self.state == "drawing" else (60, 60, 60)

        cv2.rectangle(canvas,
                      (dz_x, 10),
                      (dz_x + dz_w, h - 10),
                      dz_color, 2)

    def _draw_ui_text(self, out, w, h, board_x):
        cv2.putText(out,
                    f"SCORE: {self.engine.score}",
                    (30, 50),
                    FONT, 0.8, NEON_BLUE, 2)

        cv2.putText(out,
                    f"LINES: {self.engine.lines}",
                    (30, 85),
                    FONT, 0.7, WHITE, 1)

        if self.state == "drawing":
            cv2.putText(out,
                        "PHASE: DRAWING",
                        (w - 250, 50),
                        FONT, 0.8, NEON_GREEN, 2)

            dz_x = (w - int(w * DRAWING_ZONE_FRAC)) // 2
            self.buf.draw_on_frame(out, offset_x=dz_x)

        else:
            cv2.putText(out,
                        "PHASE: PLAYING",
                        (w - 250, 50),
                        FONT, 0.8, CYAN, 2)

    # ==========================================================
    # COLORS
    # ==========================================================
    def _get_piece_color(self, name):
        colors = {
            'I': (255, 255, 0),
            'O': (0, 255, 255),
            'T': (255, 0, 255),
            'S': (0, 255, 0),
            'Z': (0, 0, 255),
            'J': (255, 0, 0),
            'L': (0, 165, 255)
        }
        return colors.get(name, (128, 128, 128))

    # ==========================================================
    # GAME LOGIC
    # ==========================================================
    
    def _process_vision(self, cam):
        h, w = cam.shape[:2]
        dz_w = int(w * DRAWING_ZONE_FRAC)
        dz_x = (w - dz_w) // 2

        # IMPORTANT: pass BGR frame directly
        results, small_shape = self.tracker.process(cam)

        if not results or not results.multi_hand_landmarks:
            if self.buf.is_drawing:
                self.buf.end_stroke()
            return

        lm = results.multi_hand_landmarks[0]

        idx_tip = self.tracker.index_tip(lm, small_shape)
        wrist   = self.tracker.wrist(lm, small_shape)
        pinching = self.tracker.is_pinching(lm, small_shape)

        if self.state == "drawing":

            if pinching:
                if not self.buf.is_drawing:
                    self.buf.start_stroke()

                ix, iy = idx_tip
                if dz_x <= ix <= dz_x + dz_w:
                    self.buf.add_point((float(ix - dz_x), float(iy)))

            elif self.buf.is_drawing:
                self.buf.end_stroke()

            if self.buf.has_enough(15):
                img = self.buf.to_image()
                self.drawing_img = img
                self.top_pred = self.clf.predict_topk(img, k=3)

        elif self.state == "playing":

            self.gesture.update(wrist, idx_tip)
            g = self.gesture.detect()

            if g == "move_left":
                self.engine.move(0, -1)
            elif g == "move_right":
                self.engine.move(0, 1)
            elif g == "fast_drop":
                self.engine.hard_drop()

            if self.gesture.detect_rotate():
                self.engine.rotate()

    def _tick_engine(self):
            if self.state == "playing" and not self.engine.game_over:
                self.engine.update()
                if self.engine.current is None:
                    self.state = "drawing"
                    self.buf.clear()
                    self.top_pred = []

    def _handle_key(self, key):
        if key in (ord("q"), 27):
            return False

        if key == ord("r"):
            self.engine.reset()
            self.state = "drawing"

        if key in (ord(" "), 13) and self.state == "drawing":
            self._confirm()

        return True

    def _confirm(self):
        if not self.buf.has_enough(15):
            return

        img = self.buf.to_image()
        top = self.clf.predict_topk(img, k=3)

        if not top:
            return

        name, conf = top[0]

        if conf < ASK_LABEL_CONF:
            self.ask_label = True
        else:
            self._do_spawn(name)

    def _do_spawn(self, name):
        self.engine.spawn(name)
        self.state = "playing"
        self.ask_label = False
        self.buf.clear()

    def _draw_label_picker(self, frame):
        h, w = frame.shape[:2]

        cv2.rectangle(frame,
                      (w//4, h//4),
                      (3*w//4, 3*h//4),
                      (20, 20, 20), -1)

        cv2.putText(frame,
                    "LOW CONFIDENCE - SELECT PIECE",
                    (w//4 + 20, h//4 + 40),
                    FONT, 0.6, YELLOW, 2)

        for i, name in enumerate(PIECE_NAMES, start=1):
            cv2.putText(frame,
                        f"[{i}] {name}",
                        (w//4 + 40, h//4 + 80 + i*30),
                        FONT, 0.6, WHITE, 1)

        return frame


if __name__ == "__main__":
    AirTetris().run()