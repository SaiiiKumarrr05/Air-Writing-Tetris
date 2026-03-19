"""
OpenCV-based game renderer.
Draws the Tetris grid, HUD, sidebar, confidence bar, and ghost piece.
"""
from __future__ import annotations
import time
import numpy as np
import cv2
from game.pieces import COLORS_BGR, PIECE_NAMES

# ------ layout constants -----------------------------------------------
CELL       = 28
ROWS, COLS = 20, 10
SIDEBAR_W  = 290
TOPBAR_H   = 48

W_BOARD = COLS * CELL                       # 280
H_BOARD = ROWS * CELL                       # 560
W_TOTAL = W_BOARD + SIDEBAR_W               # 570
H_TOTAL = H_BOARD + TOPBAR_H               # 608

# ------ palette --------------------------------------------------------
BG         = (15,  15,  25)
GRID_LINE  = (32,  32,  52)
WHITE      = (240, 240, 240)
YELLOW     = (0,   220, 240)
RED_C      = (50,  50,  230)
GREEN_C    = (50,  200,  80)
ORANGE_C   = (0,   140, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# -----------------------------------------------------------------------
def make_frame(engine,
               drawing_img: np.ndarray | None = None,
               top_pred: list | None = None,
               state: str = "drawing") -> np.ndarray:
    """
    Returns a (H_TOTAL × W_TOTAL × 3) BGR frame ready for display.
    """
    f = np.full((H_TOTAL, W_TOTAL, 3), BG, dtype=np.uint8)
    _topbar(f, engine, state)
    _grid_lines(f)
    _locked_cells(f, engine)
    _blink(f, engine)
    _ghost(f, engine)
    _current(f, engine)
    _sidebar(f, engine, drawing_img, top_pred, state)
    return f


# -----------------------------------------------------------------------
def _cell(f, row, col, color, alpha=1.0):
    x = col * CELL
    y = TOPBAR_H + row * CELL
    if alpha < 1.0:
        ov = f.copy()
        cv2.rectangle(ov, (x+1, y+1), (x+CELL-2, y+CELL-2), color, -1)
        cv2.addWeighted(ov, alpha, f, 1.0 - alpha, 0, f)
    else:
        cv2.rectangle(f, (x+1, y+1), (x+CELL-2, y+CELL-2), color, -1)
        hi = tuple(min(c+55, 255) for c in color)
        cv2.rectangle(f, (x, y), (x+CELL, y+CELL), hi, 1)


def _topbar(f, engine, state):
    cv2.rectangle(f, (0, 0), (W_TOTAL, TOPBAR_H), (22, 22, 38), -1)
    txt = (f"SCORE {engine.score:>7}   LVL {engine.level}   "
           f"LINES {engine.lines}   T {engine.elapsed}s")
    cv2.putText(f, txt, (8, 31), FONT, 0.52, WHITE, 1, cv2.LINE_AA)
    if state == "game_over":
        cv2.putText(f, "GAME OVER", (W_BOARD//2 - 85, 38),
                    FONT, 0.80, RED_C, 2, cv2.LINE_AA)


def _grid_lines(f):
    for r in range(ROWS + 1):
        y = TOPBAR_H + r * CELL
        cv2.line(f, (0, y), (W_BOARD, y), GRID_LINE, 1)
    for c in range(COLS + 1):
        x = c * CELL
        cv2.line(f, (x, TOPBAR_H), (x, TOPBAR_H + H_BOARD), GRID_LINE, 1)


def _locked_cells(f, engine):
    for r, row in enumerate(engine.grid):
        for c, val in enumerate(row):
            if val:
                _cell(f, r, c, COLORS_BGR[val])


def _blink(f, engine):
    if not engine.blink_rows:
        return
    elapsed = time.time() - engine.blink_timer
    if elapsed > 0.35:
        engine.blink_rows = []
        return
    alpha = abs(np.sin(elapsed * 55))
    for r in engine.blink_rows:
        for c in range(COLS):
            _cell(f, r, c, (255, 255, 255), alpha=float(alpha))


def _ghost(f, engine):
    if engine.current is None:
        return
    color = COLORS_BGR[engine.current.name]
    for r, c in engine.get_ghost():
        if 0 <= r < ROWS and 0 <= c < COLS:
            _cell(f, r, c, color, alpha=0.22)


def _current(f, engine):
    if engine.current is None:
        return
    color = COLORS_BGR[engine.current.name]
    for r, c in engine.current.absolute_cells():
        if 0 <= r < ROWS and 0 <= c < COLS:
            _cell(f, r, c, color)


# -----------------------------------------------------------------------
def _sidebar(f, engine, drawing_img, top_pred, state):
    x0 = W_BOARD + 10
    y  = TOPBAR_H + 14

    cv2.putText(f, "DRAWING PREVIEW", (x0, y), FONT, 0.45, YELLOW, 1)
    y += 6

    # Drawing thumbnail
    if drawing_img is not None:
        thumb = cv2.resize(drawing_img, (100, 100))
        if thumb.ndim == 2:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        f[y:y+100, x0:x0+100] = thumb
    y += 108

    # Top predictions + confidence bar
    if top_pred:
        name, conf = top_pred[0]
        bar_color  = GREEN_C if conf >= 0.60 else ORANGE_C if conf >= 0.40 else RED_C
        col_name   = COLORS_BGR.get(name, WHITE)
        cv2.putText(f, f"Detected:  {name}", (x0, y), FONT, 0.65, col_name, 2)
        y += 26

        bar_w = int(np.clip(conf, 0, 1) * (SIDEBAR_W - 24))
        cv2.rectangle(f, (x0, y),           (x0 + bar_w,          y+16), bar_color, -1)
        cv2.rectangle(f, (x0, y),           (x0 + SIDEBAR_W - 24, y+16), WHITE,     1)
        cv2.putText(f, f"{conf*100:.0f}%",  (x0 + bar_w + 4, y+12), FONT, 0.42, WHITE, 1)
        y += 26

        # Low-confidence warning
        if conf < 0.60:
            cv2.putText(f, "! Low confidence", (x0, y), FONT, 0.44, ORANGE_C, 1)
            y += 18

        # Runner-up classes
        for rank, (n2, c2) in enumerate(top_pred[1:3], start=2):
            cv2.putText(f, f"  {rank}. {n2}  {c2*100:.0f}%",
                        (x0, y), FONT, 0.42, COLORS_BGR.get(n2, WHITE), 1)
            y += 18
    y += 10

    # Divider
    cv2.line(f, (x0, y), (x0 + SIDEBAR_W - 18, y), GRID_LINE, 1)
    y += 14

    # Controls cheatsheet
    cv2.putText(f, "CONTROLS", (x0, y), FONT, 0.48, YELLOW, 1)
    y += 18
    for key, action in [
        ("Pinch + draw", "Air-draw shape"),
        ("SPACE / ENTER", "Confirm piece"),
        ("Swipe L / R",   "Move piece"),
        ("Swipe Down",    "Hard drop"),
        ("Circle motion", "Rotate piece"),
        ("R",             "Reset game"),
        ("Q / ESC",       "Quit"),
    ]:
        cv2.putText(f, f"{key}: {action}", (x0, y), FONT, 0.38, (175, 175, 175), 1)
        y += 16
    y += 8

    # State badge
    badge_color = (0, 200, 100) if state == "drawing" else (0, 120, 255)
    badge_txt   = "DRAW MODE" if state == "drawing" else "PLAY MODE"
    cv2.rectangle(f, (x0, y), (x0 + 130, y + 22), badge_color, -1)
    cv2.putText(f, badge_txt, (x0 + 6, y + 15), FONT, 0.48, (10, 10, 10), 1)
