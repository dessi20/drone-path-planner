from __future__ import annotations
from enum import Enum
import numpy as np


class CellState(Enum):
    EMPTY = 0
    OBSTACLE = 1
    NO_FLY_ZONE = 2


class Grid:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self._cells: np.ndarray = np.zeros((rows, cols), dtype=np.uint8)

    def add_obstacle(self, r: int, c: int) -> None:
        self._cells[r, c] = CellState.OBSTACLE.value

    def add_nfz(self, r1: int, c1: int, r2: int, c2: int) -> None:
        r_min, r_max = min(r1, r2), max(r1, r2)
        c_min, c_max = min(c1, c2), max(c1, c2)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                self._cells[r, c] = CellState.NO_FLY_ZONE.value

    def cell_state(self, r: int, c: int) -> CellState:
        return CellState(int(self._cells[r, c]))

    def is_walkable(self, r: int, c: int) -> bool:
        return self._cells[r, c] == CellState.EMPTY.value

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(
        self, r: int, c: int, connectivity: int = 8
    ) -> list[tuple[int, int, float]]:
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}.")
        if connectivity == 4:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            deltas = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
        result = []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.is_walkable(nr, nc):
                cost = 2 ** 0.5 if (dr != 0 and dc != 0) else 1.0
                result.append((nr, nc, cost))
        return result


class Grid3D:
    def __init__(self, rows: int, cols: int, layers: int) -> None:
        self.rows = rows
        self.cols = cols
        self.layers = layers
        self._cells: np.ndarray = np.zeros((rows, cols, layers), dtype=np.uint8)

    def add_obstacle(self, r: int, c: int, layer: int) -> None:
        self._cells[r, c, layer] = CellState.OBSTACLE.value

    def add_nfz(self, r1: int, c1: int, r2: int, c2: int, layer: int) -> None:
        r_min, r_max = min(r1, r2), max(r1, r2)
        c_min, c_max = min(c1, c2), max(c1, c2)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                self._cells[r, c, layer] = CellState.NO_FLY_ZONE.value

    def cell_state(self, r: int, c: int, layer: int) -> CellState:
        return CellState(int(self._cells[r, c, layer]))

    def is_walkable(self, r: int, c: int, layer: int) -> bool:
        return self._cells[r, c, layer] == CellState.EMPTY.value

    def in_bounds(self, r: int, c: int, layer: int) -> bool:
        return (
            0 <= r < self.rows
            and 0 <= c < self.cols
            and 0 <= layer < self.layers
        )

    def neighbors(
        self, r: int, c: int, layer: int, connectivity: int = 26
    ) -> list[tuple[int, int, int, float]]:
        if connectivity not in (6, 26):
            raise ValueError(f"connectivity must be 6 or 26, got {connectivity!r}.")
        if connectivity == 6:
            deltas = [
                (-1, 0, 0), (1, 0, 0),
                (0, -1, 0), (0, 1, 0),
                (0, 0, -1), (0, 0, 1),
            ]
        else:
            deltas = [
                (dr, dc, dl)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                for dl in (-1, 0, 1)
                if not (dr == 0 and dc == 0 and dl == 0)
            ]
        result = []
        for dr, dc, dl in deltas:
            nr, nc, nl = r + dr, c + dc, layer + dl
            if self.in_bounds(nr, nc, nl) and self.is_walkable(nr, nc, nl):
                nonzero = sum(1 for d in (dr, dc, dl) if d != 0)
                cost = {1: 1.0, 2: 2 ** 0.5, 3: 3 ** 0.5}[nonzero]
                result.append((nr, nc, nl, cost))
        return result
