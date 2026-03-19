"""
Core Tetris engine — grid management, collision detection, scoring, gravity.
Thread-safe: all public methods acquire self._lock.
"""
from __future__ import annotations
import time
import threading
from .pieces import PIECE_ROTATIONS, SPAWN_COL, PIECE_NAMES

ROWS, COLS = 20, 10

# Points awarded for n simultaneous line clears (before level multiplier)
LINE_SCORES = [0, 100, 300, 500, 800]


class Piece:
    __slots__ = ("name", "rot", "row", "col", "cells")

    def __init__(self, name: str) -> None:
        self.name  = name
        self.rot   = 0
        self.row   = 0
        self.col   = SPAWN_COL[name]
        self.cells = PIECE_ROTATIONS[name][0]

    def absolute_cells(self) -> list[tuple[int, int]]:
        return [(self.row + r, self.col + c) for r, c in self.cells]


class TetrisEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._reset_unsafe()

    def _reset_unsafe(self) -> None:
        self.grid: list[list[str | None]] = [[None] * COLS for _ in range(ROWS)]
        self.score         = 0
        self.lines         = 0
        self.level         = 1
        self.pieces_spawned = 0
        self.combo         = 0
        self.game_over     = False
        self.current: Piece | None = None
        self.drop_interval = 0.80       # seconds between gravity ticks
        self.last_drop     = time.time()
        self.blink_rows: list[int] = []
        self.blink_timer   = 0.0
        self.last_cleared  = 0
        self._start_time   = time.time()

    def spawn(self, name: str) -> bool:
        with self._lock:
            p = Piece(name)
            self.pieces_spawned += 1
            if not self._valid(p.row, p.col, p.cells):
                self.game_over = True
                return False
            self.current = p
            return True

    def move(self, dr: int, dc: int) -> bool:
        with self._lock:
            if self.current is None:
                return False
            p = self.current
            if self._valid(p.row + dr, p.col + dc, p.cells):
                p.row += dr
                p.col += dc
                return True
            return False

    def rotate(self) -> None:
        with self._lock:
            if self.current is None:
                return
            p    = self.current
            rots = PIECE_ROTATIONS[p.name]
            nr   = (p.rot + 1) % len(rots)
            nc   = rots[nr]
            # SRS-style wall-kick attempts
            for dc in (0, -1, 1, -2, 2):
                if self._valid(p.row, p.col + dc, nc):
                    p.rot, p.cells, p.col = nr, nc, p.col + dc
                    return

    def hard_drop(self) -> None:
        with self._lock:
            if self.current is None:
                return
            p = self.current
            while self._valid(p.row + 1, p.col, p.cells):
                p.row += 1
            self._lock_unsafe()

    def update(self) -> None:
        """Call once per frame to apply gravity."""
        with self._lock:
            if self.game_over or self.current is None:
                return
            now = time.time()
            if now - self.last_drop >= self.drop_interval:
                self.last_drop = now
                p = self.current
                if self._valid(p.row + 1, p.col, p.cells):
                    p.row += 1
                else:
                    self._lock_unsafe()

    def get_ghost(self) -> list[tuple[int, int]]:
        with self._lock:
            if self.current is None:
                return []
            p, row = self.current, self.current.row
            while self._valid(row + 1, p.col, p.cells):
                row += 1
            return [(row + r, p.col + c) for r, c in p.cells]

    def stack_height(self) -> int:
        with self._lock:
            for r in range(ROWS):
                if any(self.grid[r]):
                    return ROWS - r
            return 0

    @property
    def elapsed(self) -> int:
        return int(time.time() - self._start_time)

    # ------------------------------------------------------------------
    # Internal helpers (call only while holding self._lock)
    # ------------------------------------------------------------------
    def _valid(self, row: int, col: int, cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            ar, ac = row + r, col + c
            if not (0 <= ar < ROWS and 0 <= ac < COLS):
                return False
            if self.grid[ar][ac] is not None:
                return False
        return True

    def _lock_unsafe(self) -> None:
        p = self.current
        if p is None:
            return
        for r, c in p.absolute_cells():
            if 0 <= r < ROWS and 0 <= c < COLS:
                self.grid[r][c] = p.name
        self.current = None
        self._clear_lines_unsafe()

    def _clear_lines_unsafe(self) -> None:
        full = [i for i, row in enumerate(self.grid)
                if all(c is not None for c in row)]
        self.last_cleared = len(full)
        if full:
            self.blink_rows  = list(full)
            self.blink_timer = time.time()
            n = len(full)
            self.combo += 1
            combo_mult = 1.0 + 0.5 * (self.combo - 1)
            self.score += int(LINE_SCORES[min(n, 4)] * self.level * combo_mult)
            self.lines += n
            self.level  = 1 + self.lines // 10
            self.drop_interval = max(0.08, 0.80 - 0.07 * (self.level - 1))
            for i in sorted(full, reverse=True):
                del self.grid[i]
                self.grid.insert(0, [None] * COLS)
        else:
            self.combo = 0
        self.score += 5   # small per-piece bonus
