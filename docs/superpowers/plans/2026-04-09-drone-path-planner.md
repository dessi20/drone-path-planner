# Drone Path Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that plans autonomous drone paths using A* and Dijkstra on configurable 2D/3D grids with matplotlib visualization and JSON export.

**Architecture:** Clean modular package (`drone_planner/`) with five focused modules: `grid.py` (data model), `algorithms.py` (pathfinding), `visualizer.py` (matplotlib), `exporter.py` (JSON), `cli.py` (Typer). Data flows CLI → Grid → Algorithm → Visualizer/Exporter. `Grid3D` is a sibling class to `Grid` (not a subclass) since their method signatures differ incompatibly.

**Tech Stack:** Python 3.11+, typer[all], matplotlib, numpy, rich, pytest

---

## File Map

| File | Purpose |
|---|---|
| `requirements.txt` | All runtime + dev dependencies |
| `drone_planner/__init__.py` | Package marker; exports nothing |
| `drone_planner/__main__.py` | Enables `python -m drone_planner` |
| `drone_planner/grid.py` | `CellState` enum, `Grid` (2D), `Grid3D` (3D) |
| `drone_planner/algorithms.py` | `PathResult` dataclass, `astar()`, `dijkstra()` |
| `drone_planner/visualizer.py` | `plot_2d()`, `plot_compare()`, `plot_3d()` |
| `drone_planner/exporter.py` | `export_json()` |
| `drone_planner/cli.py` | Typer app: `run`, `compare`, `demo` commands |
| `tests/__init__.py` | Package marker |
| `tests/test_grid.py` | Grid and Grid3D tests |
| `tests/test_algorithms.py` | Dijkstra, A*, 3D algorithm tests |
| `tests/test_exporter.py` | JSON export tests |
| `tests/test_visualizer.py` | Figure creation tests (Agg backend) |
| `tests/test_cli.py` | CLI command tests via typer test runner |
| `README.md` | Project docs with example outputs |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `drone_planner/__init__.py`
- Create: `drone_planner/__main__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create the requirements file**

```
# requirements.txt
typer[all]
matplotlib
numpy
rich
pytest
```

- [ ] **Step 2: Create the package files**

`drone_planner/__init__.py` — empty file, just a package marker.

`drone_planner/__main__.py`:
```python
from drone_planner.cli import app

if __name__ == "__main__":
    app()
```

`tests/__init__.py` — empty file, just a package marker.

- [ ] **Step 3: Verify Python finds the package**

Run: `python -c "import drone_planner; print('ok')"`
Expected: `ok`

---

## Task 2: 2D Grid

**Files:**
- Create: `drone_planner/grid.py`
- Create: `tests/test_grid.py`

- [ ] **Step 1: Write failing tests for Grid**

`tests/test_grid.py`:
```python
import pytest
from drone_planner.grid import Grid, CellState


def test_grid_dimensions():
    g = Grid(10, 15)
    assert g.rows == 10
    assert g.cols == 15


def test_default_cells_are_empty():
    g = Grid(5, 5)
    assert g.cell_state(0, 0) == CellState.EMPTY
    assert g.cell_state(4, 4) == CellState.EMPTY


def test_add_obstacle():
    g = Grid(5, 5)
    g.add_obstacle(2, 3)
    assert g.cell_state(2, 3) == CellState.OBSTACLE
    assert not g.is_walkable(2, 3)


def test_add_nfz_marks_all_cells():
    g = Grid(10, 10)
    g.add_nfz(1, 1, 3, 3)
    for r in range(1, 4):
        for c in range(1, 4):
            assert g.cell_state(r, c) == CellState.NO_FLY_ZONE
    assert g.cell_state(0, 0) == CellState.EMPTY


def test_add_nfz_is_not_walkable():
    g = Grid(5, 5)
    g.add_nfz(0, 0, 1, 1)
    assert not g.is_walkable(0, 0)
    assert not g.is_walkable(1, 1)


def test_is_walkable_empty():
    g = Grid(5, 5)
    assert g.is_walkable(0, 0)


def test_neighbors_4connected_center():
    g = Grid(5, 5)
    neighbors = g.neighbors(2, 2, connectivity=4)
    coords = {(r, c) for r, c, _ in neighbors}
    assert coords == {(1, 2), (3, 2), (2, 1), (2, 3)}


def test_neighbors_8connected_center():
    g = Grid(5, 5)
    neighbors = g.neighbors(2, 2, connectivity=8)
    coords = {(r, c) for r, c, _ in neighbors}
    assert len(coords) == 8
    assert (1, 1) in coords
    assert (3, 3) in coords


def test_neighbors_diagonal_cost():
    g = Grid(5, 5)
    neighbors = g.neighbors(2, 2, connectivity=8)
    cost_by_coord = {(r, c): cost for r, c, cost in neighbors}
    assert cost_by_coord[(1, 1)] == pytest.approx(2 ** 0.5)
    assert cost_by_coord[(1, 2)] == pytest.approx(1.0)


def test_neighbors_excludes_out_of_bounds():
    g = Grid(5, 5)
    neighbors = g.neighbors(0, 0, connectivity=8)
    coords = {(r, c) for r, c, _ in neighbors}
    assert coords == {(0, 1), (1, 0), (1, 1)}


def test_neighbors_excludes_obstacles():
    g = Grid(5, 5)
    g.add_obstacle(1, 2)
    neighbors = g.neighbors(2, 2, connectivity=4)
    coords = {(r, c) for r, c, _ in neighbors}
    assert (1, 2) not in coords


def test_neighbors_excludes_nfz():
    g = Grid(5, 5)
    g.add_nfz(1, 2, 1, 2)
    neighbors = g.neighbors(2, 2, connectivity=4)
    coords = {(r, c) for r, c, _ in neighbors}
    assert (1, 2) not in coords
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_grid.py -v`
Expected: Multiple failures with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `grid.py`**

`drone_planner/grid.py`:
```python
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
        self._cells: np.ndarray = np.full(
            (rows, cols), CellState.EMPTY, dtype=object
        )

    def add_obstacle(self, r: int, c: int) -> None:
        self._cells[r, c] = CellState.OBSTACLE

    def add_nfz(self, r1: int, c1: int, r2: int, c2: int) -> None:
        r_min, r_max = min(r1, r2), max(r1, r2)
        c_min, c_max = min(c1, c2), max(c1, c2)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                self._cells[r, c] = CellState.NO_FLY_ZONE

    def cell_state(self, r: int, c: int) -> CellState:
        return self._cells[r, c]

    def is_walkable(self, r: int, c: int) -> bool:
        return self._cells[r, c] == CellState.EMPTY

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(
        self, r: int, c: int, connectivity: int = 8
    ) -> list[tuple[int, int, float]]:
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
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `pytest tests/test_grid.py -v`
Expected: All tests PASS

---

## Task 3: 3D Grid

**Files:**
- Modify: `drone_planner/grid.py` (append `Grid3D` class)
- Modify: `tests/test_grid.py` (append Grid3D tests)

- [ ] **Step 1: Write failing tests for Grid3D**

Append to `tests/test_grid.py`:
```python
from drone_planner.grid import Grid3D


def test_grid3d_dimensions():
    g = Grid3D(5, 5, 3)
    assert g.rows == 5
    assert g.cols == 5
    assert g.layers == 3


def test_grid3d_default_cells_empty():
    g = Grid3D(3, 3, 2)
    assert g.cell_state(0, 0, 0) == CellState.EMPTY


def test_grid3d_add_obstacle():
    g = Grid3D(5, 5, 3)
    g.add_obstacle(1, 1, 1)
    assert g.cell_state(1, 1, 1) == CellState.OBSTACLE
    assert not g.is_walkable(1, 1, 1)
    assert g.is_walkable(1, 1, 0)


def test_grid3d_add_nfz():
    g = Grid3D(5, 5, 3)
    g.add_nfz(0, 0, 1, 1, 2)
    for r in range(2):
        for c in range(2):
            assert g.cell_state(r, c, 2) == CellState.NO_FLY_ZONE
    assert g.cell_state(0, 0, 0) == CellState.EMPTY


def test_grid3d_neighbors_6connected():
    g = Grid3D(5, 5, 3)
    neighbors = g.neighbors(2, 2, 1, connectivity=6)
    coords = {(r, c, l) for r, c, l, _ in neighbors}
    assert coords == {(1, 2, 1), (3, 2, 1), (2, 1, 1), (2, 3, 1), (2, 2, 0), (2, 2, 2)}


def test_grid3d_neighbors_26connected_count():
    g = Grid3D(5, 5, 5)
    neighbors = g.neighbors(2, 2, 2, connectivity=26)
    assert len(neighbors) == 26


def test_grid3d_neighbor_face_cost():
    g = Grid3D(5, 5, 3)
    neighbors = g.neighbors(2, 2, 1, connectivity=26)
    cost_by_coord = {(r, c, l): cost for r, c, l, cost in neighbors}
    assert cost_by_coord[(1, 2, 1)] == pytest.approx(1.0)


def test_grid3d_neighbor_edge_cost():
    g = Grid3D(5, 5, 3)
    neighbors = g.neighbors(2, 2, 1, connectivity=26)
    cost_by_coord = {(r, c, l): cost for r, c, l, cost in neighbors}
    assert cost_by_coord[(1, 1, 1)] == pytest.approx(2 ** 0.5)


def test_grid3d_neighbor_corner_cost():
    g = Grid3D(5, 5, 3)
    neighbors = g.neighbors(2, 2, 1, connectivity=26)
    cost_by_coord = {(r, c, l): cost for r, c, l, cost in neighbors}
    assert cost_by_coord[(1, 1, 0)] == pytest.approx(3 ** 0.5)


def test_grid3d_neighbors_excludes_obstacles():
    g = Grid3D(5, 5, 3)
    g.add_obstacle(1, 2, 1)
    neighbors = g.neighbors(2, 2, 1, connectivity=6)
    coords = {(r, c, l) for r, c, l, _ in neighbors}
    assert (1, 2, 1) not in coords
```

- [ ] **Step 2: Run tests to confirm new tests fail**

Run: `pytest tests/test_grid.py -v -k "grid3d"`
Expected: Failures with `ImportError: cannot import name 'Grid3D'`

- [ ] **Step 3: Implement `Grid3D` — append to `drone_planner/grid.py`**

```python
class Grid3D:
    def __init__(self, rows: int, cols: int, layers: int) -> None:
        self.rows = rows
        self.cols = cols
        self.layers = layers
        self._cells: np.ndarray = np.full(
            (rows, cols, layers), CellState.EMPTY, dtype=object
        )

    def add_obstacle(self, r: int, c: int, layer: int) -> None:
        self._cells[r, c, layer] = CellState.OBSTACLE

    def add_nfz(self, r1: int, c1: int, r2: int, c2: int, layer: int) -> None:
        r_min, r_max = min(r1, r2), max(r1, r2)
        c_min, c_max = min(c1, c2), max(c1, c2)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                self._cells[r, c, layer] = CellState.NO_FLY_ZONE

    def cell_state(self, r: int, c: int, layer: int) -> CellState:
        return self._cells[r, c, layer]

    def is_walkable(self, r: int, c: int, layer: int) -> bool:
        return self._cells[r, c, layer] == CellState.EMPTY

    def in_bounds(self, r: int, c: int, layer: int) -> bool:
        return (
            0 <= r < self.rows
            and 0 <= c < self.cols
            and 0 <= layer < self.layers
        )

    def neighbors(
        self, r: int, c: int, layer: int, connectivity: int = 26
    ) -> list[tuple[int, int, int, float]]:
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
```

- [ ] **Step 4: Run all grid tests**

Run: `pytest tests/test_grid.py -v`
Expected: All tests PASS

---

## Task 4: PathResult Dataclass + Dijkstra 2D

**Files:**
- Create: `drone_planner/algorithms.py`
- Create: `tests/test_algorithms.py`

- [ ] **Step 1: Write failing tests for Dijkstra 2D**

`tests/test_algorithms.py`:
```python
import pytest
from drone_planner.grid import Grid, Grid3D
from drone_planner.algorithms import dijkstra, astar, PathResult


# --- helpers ---

def open_grid(n: int = 10) -> Grid:
    return Grid(n, n)


def blocked_grid() -> Grid:
    """5x5 grid with column 2 fully blocked — goal at (0,4) is unreachable from (0,0)."""
    g = Grid(5, 5)
    for r in range(5):
        g.add_obstacle(r, 2)
    return g


# --- Dijkstra 2D ---

def test_dijkstra_finds_path_open_grid():
    result = dijkstra(open_grid(), (0, 0), (9, 9))
    assert result.found
    assert len(result.path) > 0


def test_dijkstra_no_path_when_blocked():
    result = dijkstra(blocked_grid(), (0, 0), (0, 4))
    assert not result.found
    assert result.path == []


def test_dijkstra_path_starts_at_start():
    result = dijkstra(open_grid(), (0, 0), (9, 9))
    assert result.path[0] == (0, 0)


def test_dijkstra_path_ends_at_goal():
    result = dijkstra(open_grid(), (0, 0), (9, 9))
    assert result.path[-1] == (9, 9)


def test_dijkstra_has_stats():
    result = dijkstra(open_grid(), (0, 0), (9, 9))
    assert result.nodes_explored > 0
    assert result.compute_time_ms >= 0.0
    assert result.path_length > 0.0
    assert result.algorithm_name == "dijkstra"


def test_dijkstra_4connected_path_length():
    """On open 5x5 grid, 4-connected Dijkstra from (0,0) to (4,4) = 8 cardinal steps."""
    g = Grid(5, 5)
    result = dijkstra(g, (0, 0), (4, 4), connectivity=4)
    assert result.found
    assert result.path_length == pytest.approx(8.0)


def test_dijkstra_8connected_path_length():
    """On open 5x5 grid, 8-connected Dijkstra from (0,0) to (4,4) = 4 diagonal steps."""
    g = Grid(5, 5)
    result = dijkstra(g, (0, 0), (4, 4), connectivity=8)
    assert result.found
    assert result.path_length == pytest.approx(4 * (2 ** 0.5))


def test_dijkstra_explored_nodes_populated():
    result = dijkstra(open_grid(), (0, 0), (9, 9))
    assert len(result.explored_nodes) > 0


def test_dijkstra_start_equals_goal():
    g = Grid(5, 5)
    result = dijkstra(g, (2, 2), (2, 2))
    assert result.found
    assert result.path == [(2, 2)]
    assert result.path_length == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_algorithms.py -v -k "dijkstra"`
Expected: Failures with `ImportError`

- [ ] **Step 3: Implement `PathResult` + `dijkstra()` in `algorithms.py`**

`drone_planner/algorithms.py`:
```python
from __future__ import annotations
import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Union

from drone_planner.grid import Grid, Grid3D


@dataclass
class PathResult:
    algorithm_name: str
    path: list[tuple]
    path_length: float
    nodes_explored: int
    compute_time_ms: float
    found: bool
    explored_nodes: list[tuple] = field(default_factory=list)


def _reconstruct_path(came_from: dict, current: tuple) -> list[tuple]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _get_neighbors_2d(
    grid: Grid, node: tuple, connectivity: int
) -> list[tuple[tuple, float]]:
    r, c = node
    return [((nr, nc), cost) for nr, nc, cost in grid.neighbors(r, c, connectivity)]


def dijkstra(
    grid: Union[Grid, Grid3D],
    start: tuple,
    goal: tuple,
    connectivity: int = 8,
) -> PathResult:
    start_time = time.perf_counter()
    is_3d = isinstance(grid, Grid3D)

    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000
        return PathResult("dijkstra", [start], 0.0, 1, elapsed, True, [start])

    open_heap: list[tuple] = [(0.0, 0, start)]
    came_from: dict = {}
    dist: dict = {start: 0.0}
    explored: set = set()
    counter = 1

    while open_heap:
        d, _, current = heapq.heappop(open_heap)
        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            elapsed = (time.perf_counter() - start_time) * 1000
            return PathResult(
                "dijkstra", path, dist[goal], len(explored), elapsed, True,
                list(explored),
            )

        if is_3d:
            r, c, layer = current
            raw = grid.neighbors(r, c, layer, connectivity=connectivity)
            neighbors = [((nr, nc, nl), cost) for nr, nc, nl, cost in raw]
        else:
            r, c = current
            raw = grid.neighbors(r, c, connectivity=connectivity)
            neighbors = [((nr, nc), cost) for nr, nc, cost in raw]

        for neighbor, cost in neighbors:
            tentative = dist[current] + cost
            if neighbor not in dist or tentative < dist[neighbor]:
                dist[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative, counter, neighbor))
                counter += 1

    elapsed = (time.perf_counter() - start_time) * 1000
    return PathResult("dijkstra", [], 0.0, len(explored), elapsed, False, list(explored))
```

- [ ] **Step 4: Run Dijkstra tests**

Run: `pytest tests/test_algorithms.py -v -k "dijkstra"`
Expected: All PASS

---

## Task 5: A* 2D

**Files:**
- Modify: `drone_planner/algorithms.py` (append `_heuristic()` + `astar()`)
- Modify: `tests/test_algorithms.py` (append A* tests)

- [ ] **Step 1: Write failing A* tests**

Append to `tests/test_algorithms.py`:
```python
# --- A* 2D ---

def test_astar_finds_path_open_grid():
    result = astar(open_grid(), (0, 0), (9, 9))
    assert result.found
    assert len(result.path) > 0


def test_astar_no_path_when_blocked():
    result = astar(blocked_grid(), (0, 0), (0, 4))
    assert not result.found
    assert result.path == []


def test_astar_path_endpoints():
    result = astar(open_grid(), (0, 0), (9, 9))
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (9, 9)


def test_astar_algorithm_name():
    result = astar(Grid(5, 5), (0, 0), (4, 4))
    assert result.algorithm_name == "astar"


def test_astar_optimal_vs_dijkstra():
    """A* path length must equal Dijkstra's on the same grid (admissible heuristic)."""
    g = Grid(15, 15)
    for r in range(5, 10):
        g.add_obstacle(r, 7)
    a = astar(g, (0, 0), (14, 14))
    d = dijkstra(g, (0, 0), (14, 14))
    assert a.found and d.found
    assert a.path_length == pytest.approx(d.path_length, rel=1e-6)


def test_astar_explores_fewer_nodes_than_dijkstra():
    g = Grid(20, 20)
    a = astar(g, (0, 0), (19, 19))
    d = dijkstra(g, (0, 0), (19, 19))
    assert a.nodes_explored <= d.nodes_explored


def test_astar_manhattan_heuristic_4connected():
    g = Grid(5, 5)
    result = astar(g, (0, 0), (4, 4), connectivity=4, heuristic="manhattan")
    assert result.found
    assert result.path_length == pytest.approx(8.0)


def test_astar_euclidean_heuristic():
    g = Grid(10, 10)
    result = astar(g, (0, 0), (9, 9), heuristic="euclidean")
    assert result.found


def test_astar_start_equals_goal():
    g = Grid(5, 5)
    result = astar(g, (2, 2), (2, 2))
    assert result.found
    assert result.path == [(2, 2)]
    assert result.path_length == pytest.approx(0.0)


def test_astar_explored_nodes_populated():
    result = astar(open_grid(), (0, 0), (9, 9))
    assert len(result.explored_nodes) > 0
```

- [ ] **Step 2: Run to confirm A* tests fail**

Run: `pytest tests/test_algorithms.py -v -k "astar"`
Expected: Failures because `astar` not yet defined

- [ ] **Step 3: Implement `_heuristic()` and `astar()` — append to `drone_planner/algorithms.py`**

```python
def _heuristic(a: tuple, b: tuple, heuristic: str) -> float:
    if len(a) == 2:
        r1, c1 = a
        r2, c2 = b
        if heuristic == "manhattan":
            return abs(r1 - r2) + abs(c1 - c2)
        elif heuristic == "chebyshev":
            return max(abs(r1 - r2), abs(c1 - c2))
        else:
            return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
    else:
        r1, c1, l1 = a
        r2, c2, l2 = b
        if heuristic == "manhattan":
            return abs(r1 - r2) + abs(c1 - c2) + abs(l1 - l2)
        elif heuristic == "chebyshev":
            return max(abs(r1 - r2), abs(c1 - c2), abs(l1 - l2))
        else:
            return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2 + (l1 - l2) ** 2)


def astar(
    grid: Union[Grid, Grid3D],
    start: tuple,
    goal: tuple,
    connectivity: int = 8,
    heuristic: str = "chebyshev",
) -> PathResult:
    start_time = time.perf_counter()
    is_3d = isinstance(grid, Grid3D)

    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000
        return PathResult("astar", [start], 0.0, 1, elapsed, True, [start])

    h0 = _heuristic(start, goal, heuristic)
    open_heap: list[tuple] = [(h0, 0, start)]
    came_from: dict = {}
    g_score: dict = {start: 0.0}
    explored: set = set()
    counter = 1

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            elapsed = (time.perf_counter() - start_time) * 1000
            return PathResult(
                "astar", path, g_score[goal], len(explored), elapsed, True,
                list(explored),
            )

        if is_3d:
            r, c, layer = current
            raw = grid.neighbors(r, c, layer, connectivity=connectivity)
            neighbors = [((nr, nc, nl), cost) for nr, nc, nl, cost in raw]
        else:
            r, c = current
            raw = grid.neighbors(r, c, connectivity=connectivity)
            neighbors = [((nr, nc), cost) for nr, nc, cost in raw]

        for neighbor, cost in neighbors:
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + _heuristic(neighbor, goal, heuristic)
                heapq.heappush(open_heap, (f, counter, neighbor))
                counter += 1

    elapsed = (time.perf_counter() - start_time) * 1000
    return PathResult("astar", [], 0.0, len(explored), elapsed, False, list(explored))
```

- [ ] **Step 4: Run all algorithm tests so far**

Run: `pytest tests/test_algorithms.py -v`
Expected: All PASS

---

## Task 6: 3D Pathfinding

**Files:**
- No changes to `drone_planner/algorithms.py` — the 2D/3D dispatch is already in `dijkstra()` and `astar()` via `isinstance(grid, Grid3D)`
- Modify: `tests/test_algorithms.py` (append 3D tests)

- [ ] **Step 1: Write failing 3D algorithm tests**

Append to `tests/test_algorithms.py`:
```python
# --- 3D algorithms ---

def test_dijkstra_3d_finds_path():
    g = Grid3D(5, 5, 3)
    result = dijkstra(g, (0, 0, 0), (4, 4, 2))
    assert result.found
    assert result.path[0] == (0, 0, 0)
    assert result.path[-1] == (4, 4, 2)


def test_astar_3d_finds_path():
    g = Grid3D(5, 5, 3)
    result = astar(g, (0, 0, 0), (4, 4, 2), heuristic="euclidean")
    assert result.found
    assert result.path[0] == (0, 0, 0)
    assert result.path[-1] == (4, 4, 2)


def test_3d_path_crosses_layers():
    """Path from layer 0 to layer 2 must visit at least 2 distinct layers."""
    g = Grid3D(3, 3, 3)
    result = dijkstra(g, (0, 0, 0), (0, 0, 2), connectivity=6)
    assert result.found
    layers_visited = {p[2] for p in result.path}
    assert len(layers_visited) > 1


def test_3d_no_path_when_all_exits_blocked():
    """Fully block layer 1 — goal on layer 1 is unreachable from layer 0."""
    g = Grid3D(3, 3, 2)
    for r in range(3):
        for c in range(3):
            g.add_obstacle(r, c, 1)
    result = dijkstra(g, (0, 0, 0), (2, 2, 1))
    assert not result.found


def test_3d_6connected_does_not_allow_diagonal():
    """With 6-connectivity, going from (0,0,0) to (1,1,0) should require 2 steps."""
    g = Grid3D(5, 5, 3)
    result = dijkstra(g, (0, 0, 0), (1, 1, 0), connectivity=6)
    assert result.found
    assert result.path_length == pytest.approx(2.0)


def test_3d_26connected_allows_diagonal():
    """With 26-connectivity, going from (0,0,0) to (1,1,0) is one diagonal step."""
    g = Grid3D(5, 5, 3)
    result = dijkstra(g, (0, 0, 0), (1, 1, 0), connectivity=26)
    assert result.found
    assert result.path_length == pytest.approx(2 ** 0.5)


def test_3d_astar_optimal_vs_dijkstra():
    g = Grid3D(8, 8, 4)
    g.add_obstacle(3, 3, 1)
    g.add_obstacle(4, 4, 2)
    a = astar(g, (0, 0, 0), (7, 7, 3), heuristic="euclidean")
    d = dijkstra(g, (0, 0, 0), (7, 7, 3))
    assert a.found and d.found
    assert a.path_length == pytest.approx(d.path_length, rel=1e-6)
```

- [ ] **Step 2: Run 3D tests**

Run: `pytest tests/test_algorithms.py -v -k "3d"`
Expected: All PASS (the dispatch is already implemented in Task 4–5)

- [ ] **Step 3: Run full algorithm test suite**

Run: `pytest tests/test_algorithms.py -v`
Expected: All tests PASS

---

## Task 7: JSON Exporter

**Files:**
- Create: `drone_planner/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write failing exporter tests**

`tests/test_exporter.py`:
```python
import json
import pytest
from pathlib import Path
from drone_planner.grid import Grid, Grid3D
from drone_planner.algorithms import dijkstra, astar
from drone_planner.exporter import export_json


def test_export_creates_file(tmp_path):
    g = Grid(5, 5)
    result = dijkstra(g, (0, 0), (4, 4))
    out = tmp_path / "results.json"
    export_json(g, [result], out)
    assert out.exists()


def test_export_top_level_keys(tmp_path):
    g = Grid(5, 5)
    result = dijkstra(g, (0, 0), (4, 4))
    out = tmp_path / "results.json"
    export_json(g, [result], out)
    data = json.loads(out.read_text())
    assert "grid" in data
    assert "algorithms" in data


def test_export_grid_2d_info(tmp_path):
    g = Grid(8, 12)
    result = dijkstra(g, (0, 0), (7, 11))
    out = tmp_path / "r.json"
    export_json(g, [result], out)
    data = json.loads(out.read_text())
    assert data["grid"]["rows"] == 8
    assert data["grid"]["cols"] == 12
    assert data["grid"]["layers"] == 1


def test_export_grid_3d_info(tmp_path):
    g = Grid3D(5, 5, 4)
    result = dijkstra(g, (0, 0, 0), (4, 4, 3))
    out = tmp_path / "r.json"
    export_json(g, [result], out)
    data = json.loads(out.read_text())
    assert data["grid"]["layers"] == 4


def test_export_algorithm_fields(tmp_path):
    g = Grid(5, 5)
    result = dijkstra(g, (0, 0), (4, 4))
    out = tmp_path / "r.json"
    export_json(g, [result], out)
    data = json.loads(out.read_text())
    algo = data["algorithms"][0]
    assert algo["name"] == "dijkstra"
    assert isinstance(algo["path"], list)
    assert isinstance(algo["path_length"], float)
    assert isinstance(algo["nodes_explored"], int)
    assert isinstance(algo["compute_time_ms"], float)
    assert algo["found"] is True


def test_export_path_roundtrip(tmp_path):
    g = Grid(5, 5)
    result = dijkstra(g, (0, 0), (4, 4))
    out = tmp_path / "r.json"
    export_json(g, [result], out)
    data = json.loads(out.read_text())
    path = data["algorithms"][0]["path"]
    assert path[0] == [0, 0]
    assert path[-1] == [4, 4]


def test_export_multiple_results(tmp_path):
    g = Grid(5, 5)
    r1 = dijkstra(g, (0, 0), (4, 4))
    r2 = astar(g, (0, 0), (4, 4))
    out = tmp_path / "r.json"
    export_json(g, [r1, r2], out)
    data = json.loads(out.read_text())
    assert len(data["algorithms"]) == 2
    names = {a["name"] for a in data["algorithms"]}
    assert names == {"dijkstra", "astar"}


def test_export_no_path_result(tmp_path):
    g = Grid(5, 5)
    for r in range(5):
        g.add_obstacle(r, 2)
    result = dijkstra(g, (0, 0), (0, 4))
    out = tmp_path / "r.json"
    export_json(g, [result], out)
    data = json.loads(out.read_text())
    assert data["algorithms"][0]["found"] is False
    assert data["algorithms"][0]["path"] == []
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_exporter.py -v`
Expected: Failures with `ImportError`

- [ ] **Step 3: Implement `drone_planner/exporter.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Union

from drone_planner.algorithms import PathResult
from drone_planner.grid import Grid, Grid3D


def export_json(
    grid: Union[Grid, Grid3D],
    results: list[PathResult],
    output_path: Union[str, Path],
) -> None:
    data = {
        "grid": _grid_info(grid),
        "algorithms": [_result_to_dict(r) for r in results],
    }
    Path(output_path).write_text(json.dumps(data, indent=2))


def _grid_info(grid: Union[Grid, Grid3D]) -> dict:
    info: dict = {"rows": grid.rows, "cols": grid.cols}
    info["layers"] = grid.layers if isinstance(grid, Grid3D) else 1
    return info


def _result_to_dict(result: PathResult) -> dict:
    return {
        "name": result.algorithm_name,
        "path": [list(p) for p in result.path],
        "path_length": round(result.path_length, 4),
        "nodes_explored": result.nodes_explored,
        "compute_time_ms": round(result.compute_time_ms, 4),
        "found": result.found,
    }
```

- [ ] **Step 4: Run exporter tests**

Run: `pytest tests/test_exporter.py -v`
Expected: All PASS

---

## Task 8: 2D Visualizer

**Files:**
- Create: `drone_planner/visualizer.py`
- Create: `tests/test_visualizer.py`

- [ ] **Step 1: Write failing visualizer tests**

`tests/test_visualizer.py`:
```python
import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib import

import matplotlib.pyplot as plt
import pytest
from drone_planner.grid import Grid, Grid3D
from drone_planner.algorithms import dijkstra, astar
from drone_planner.visualizer import plot_2d, plot_compare, plot_3d


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


def test_plot_2d_returns_figure():
    g = Grid(5, 5)
    result = dijkstra(g, (0, 0), (4, 4))
    fig = plot_2d(g, result, show=False)
    assert fig is not None


def test_plot_2d_has_one_axes():
    g = Grid(5, 5)
    result = dijkstra(g, (0, 0), (4, 4))
    fig = plot_2d(g, result, show=False)
    assert len(fig.axes) == 1


def test_plot_2d_with_obstacles_and_nfz():
    g = Grid(10, 10)
    g.add_obstacle(3, 3)
    g.add_nfz(5, 5, 7, 7)
    result = dijkstra(g, (0, 0), (9, 9))
    fig = plot_2d(g, result, show=False)
    assert len(fig.axes) == 1


def test_plot_2d_no_path_does_not_raise():
    g = Grid(5, 5)
    for r in range(5):
        g.add_obstacle(r, 2)
    result = dijkstra(g, (0, 0), (0, 4))
    fig = plot_2d(g, result, show=False)
    assert fig is not None


def test_plot_compare_returns_figure():
    g = Grid(5, 5)
    r1 = dijkstra(g, (0, 0), (4, 4))
    r2 = astar(g, (0, 0), (4, 4))
    fig = plot_compare(g, [r1, r2], show=False)
    assert fig is not None


def test_plot_compare_has_two_axes():
    g = Grid(5, 5)
    r1 = dijkstra(g, (0, 0), (4, 4))
    r2 = astar(g, (0, 0), (4, 4))
    fig = plot_compare(g, [r1, r2], show=False)
    assert len(fig.axes) == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_visualizer.py -v -k "not 3d"`
Expected: Failures with `ImportError`

- [ ] **Step 3: Implement `plot_2d()` and `plot_compare()` in `drone_planner/visualizer.py`**

```python
from __future__ import annotations
from typing import Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from drone_planner.algorithms import PathResult
from drone_planner.grid import CellState, Grid, Grid3D


def plot_2d(
    grid: Grid,
    result: PathResult,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Figure:
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    _draw_grid_2d(ax, grid)
    _draw_explored_2d(ax, result)
    _draw_path_2d(ax, result)

    default_title = (
        f"{result.algorithm_name.upper()} | "
        f"length={result.path_length:.2f}  "
        f"nodes={result.nodes_explored}  "
        f"time={result.compute_time_ms:.2f}ms"
    )
    ax.set_title(title or default_title, fontsize=9)
    ax.set_xlim(-0.5, grid.cols - 0.5)
    ax.set_ylim(grid.rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if show:
        plt.show()
    return fig


def plot_compare(
    grid: Grid,
    results: list[PathResult],
    show: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 8))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        plot_2d(grid, result, ax=ax, show=False)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _draw_grid_2d(ax: plt.Axes, grid: Grid) -> None:
    for r in range(grid.rows):
        for c in range(grid.cols):
            state = grid.cell_state(r, c)
            if state == CellState.OBSTACLE:
                facecolor = "#444444"
                hatch = None
                edgecolor = "#222222"
            elif state == CellState.NO_FLY_ZONE:
                facecolor = "#ffaaaa"
                hatch = "////"
                edgecolor = "#cc0000"
            else:
                facecolor = "white"
                hatch = None
                edgecolor = "#cccccc"
            rect = mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.4,
                hatch=hatch,
            )
            ax.add_patch(rect)


def _draw_explored_2d(ax: plt.Axes, result: PathResult) -> None:
    path_set = set(map(tuple, result.path))
    for node in result.explored_nodes:
        if node not in path_set:
            r, c = node[0], node[1]
            rect = mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor="#add8e6",
                edgecolor="none",
                alpha=0.6,
            )
            ax.add_patch(rect)


def _draw_path_2d(ax: plt.Axes, result: PathResult) -> None:
    if not result.path:
        return
    rows = [p[0] for p in result.path]
    cols = [p[1] for p in result.path]
    ax.plot(cols, rows, color="#00aa00", linewidth=2, zorder=3)
    ax.scatter(cols, rows, color="#00aa00", s=20, zorder=4)
    ax.scatter(cols[0], rows[0], color="#00cc00", marker="*", s=300, zorder=5, label="Start")
    ax.scatter(cols[-1], rows[-1], color="#cc0000", marker="*", s=300, zorder=5, label="Goal")
```

- [ ] **Step 4: Run 2D visualizer tests**

Run: `pytest tests/test_visualizer.py -v -k "not 3d"`
Expected: All PASS

---

## Task 9: 3D Visualizer

**Files:**
- Modify: `drone_planner/visualizer.py` (append `plot_3d()`)
- Modify: `tests/test_visualizer.py` (append 3D tests)

- [ ] **Step 1: Write failing 3D visualizer tests**

Append to `tests/test_visualizer.py`:
```python
from mpl_toolkits.mplot3d import Axes3D


def test_plot_3d_returns_figure():
    g = Grid3D(5, 5, 3)
    result = dijkstra(g, (0, 0, 0), (4, 4, 2))
    fig = plot_3d(g, result, show=False)
    assert fig is not None


def test_plot_3d_has_3d_axes():
    g = Grid3D(5, 5, 3)
    result = dijkstra(g, (0, 0, 0), (4, 4, 2))
    fig = plot_3d(g, result, show=False)
    assert isinstance(fig.axes[0], Axes3D)


def test_plot_3d_with_obstacles():
    g = Grid3D(5, 5, 3)
    g.add_obstacle(2, 2, 1)
    g.add_nfz(0, 0, 1, 1, 2)
    result = dijkstra(g, (0, 0, 0), (4, 4, 2))
    fig = plot_3d(g, result, show=False)
    assert fig is not None


def test_plot_3d_no_path_does_not_raise():
    g = Grid3D(3, 3, 2)
    for r in range(3):
        for c in range(3):
            g.add_obstacle(r, c, 1)
    result = dijkstra(g, (0, 0, 0), (2, 2, 1))
    fig = plot_3d(g, result, show=False)
    assert fig is not None
```

- [ ] **Step 2: Run to confirm 3D tests fail**

Run: `pytest tests/test_visualizer.py -v -k "3d"`
Expected: Failures with `ImportError: cannot import name 'plot_3d'`

- [ ] **Step 3: Implement `plot_3d()` — append to `drone_planner/visualizer.py`**

```python
def plot_3d(
    grid: Grid3D,
    result: PathResult,
    show: bool = True,
) -> plt.Figure:
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 registers projection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    filled = np.zeros((grid.cols, grid.rows, grid.layers), dtype=bool)
    face_colors = np.empty((grid.cols, grid.rows, grid.layers, 4))

    for r in range(grid.rows):
        for c in range(grid.cols):
            for layer in range(grid.layers):
                state = grid.cell_state(r, c, layer)
                if state == CellState.OBSTACLE:
                    filled[c, r, layer] = True
                    face_colors[c, r, layer] = (0.27, 0.27, 0.27, 0.85)
                elif state == CellState.NO_FLY_ZONE:
                    filled[c, r, layer] = True
                    face_colors[c, r, layer] = (0.9, 0.2, 0.2, 0.6)

    if filled.any():
        ax.voxels(filled, facecolors=face_colors, edgecolor="none")

    if result.path:
        xs = [p[1] for p in result.path]
        ys = [p[0] for p in result.path]
        zs = [p[2] for p in result.path]
        ax.plot(xs, ys, zs, color="#00aa00", linewidth=2, zorder=5)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color="#00cc00", marker="*", s=200, zorder=6)
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="#cc0000", marker="*", s=200, zorder=6)

    ax.set_xlabel("Col")
    ax.set_ylabel("Row")
    ax.set_zlabel("Layer")
    ax.view_init(elev=30, azim=45)
    ax.set_title(
        f"{result.algorithm_name.upper()} 3D | "
        f"length={result.path_length:.2f}  "
        f"nodes={result.nodes_explored}  "
        f"time={result.compute_time_ms:.2f}ms",
        fontsize=9,
    )

    if show:
        plt.show()
    return fig
```

- [ ] **Step 4: Run all visualizer tests**

Run: `pytest tests/test_visualizer.py -v`
Expected: All PASS

---

## Task 10: CLI — `run` and `compare` Commands

**Files:**
- Create: `drone_planner/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests for `run` and `compare`**

`tests/test_cli.py`:
```python
import json
from typer.testing import CliRunner
from drone_planner.cli import app

runner = CliRunner()


def test_run_default_exits_zero():
    result = runner.invoke(app, ["run", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_run_dijkstra():
    result = runner.invoke(app, ["run", "--algorithm", "dijkstra", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_run_with_obstacle():
    result = runner.invoke(app, ["run", "--obstacle", "5,5", "--obstacle", "5,6", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_run_with_nfz():
    result = runner.invoke(app, ["run", "--nfz", "3,3,6,6", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_run_4connected():
    result = runner.invoke(app, ["run", "--connectivity", "4", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_run_exports_json(tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(app, ["run", "--export", str(out), "--no-viz"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert "grid" in data
    assert "algorithms" in data


def test_run_3d_mode():
    result = runner.invoke(app, [
        "run", "--rows", "5", "--cols", "5", "--layers", "3",
        "--start", "0,0,0", "--goal", "4,4,2", "--no-viz",
    ])
    assert result.exit_code == 0, result.output


def test_run_3d_with_obstacle():
    result = runner.invoke(app, [
        "run", "--rows", "5", "--cols", "5", "--layers", "3",
        "--start", "0,0,0", "--goal", "4,4,2",
        "--obstacle", "2,2,1", "--no-viz",
    ])
    assert result.exit_code == 0, result.output


def test_run_3d_with_nfz():
    result = runner.invoke(app, [
        "run", "--rows", "5", "--cols", "5", "--layers", "3",
        "--start", "0,0,0", "--goal", "4,4,2",
        "--nfz", "1,1,2,2,1", "--no-viz",
    ])
    assert result.exit_code == 0, result.output


def test_compare_exits_zero():
    result = runner.invoke(app, ["compare", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_compare_outputs_stats_table():
    result = runner.invoke(app, ["compare", "--no-viz"])
    assert result.exit_code == 0
    assert "astar" in result.output.lower() or "dijkstra" in result.output.lower()


def test_compare_exports_json(tmp_path):
    out = tmp_path / "cmp.json"
    result = runner.invoke(app, ["compare", "--export", str(out), "--no-viz"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data["algorithms"]) == 2
```

- [ ] **Step 2: Run to confirm tests fail**

Run: `pytest tests/test_cli.py -v -k "run or compare"`
Expected: Failures with `ImportError`

- [ ] **Step 3: Implement `drone_planner/cli.py` with `run` and `compare`**

```python
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from drone_planner.algorithms import PathResult, astar, dijkstra
from drone_planner.exporter import export_json
from drone_planner.grid import Grid, Grid3D
from drone_planner.visualizer import plot_2d, plot_compare, plot_3d

app = typer.Typer(name="drone-planner", help="Autonomous drone path planning simulation.")
console = Console()


# --- coordinate parsers ---

def _parse_coord(s: str, is_3d: bool) -> tuple:
    parts = [int(x) for x in s.split(",")]
    if is_3d:
        if len(parts) != 3:
            raise typer.BadParameter(f"3D coord needs r,c,l — got '{s}'")
        return tuple(parts)
    else:
        if len(parts) != 2:
            raise typer.BadParameter(f"2D coord needs r,c — got '{s}'")
        return tuple(parts)


def _parse_nfz(s: str, is_3d: bool) -> tuple:
    parts = [int(x) for x in s.split(",")]
    if is_3d:
        if len(parts) != 5:
            raise typer.BadParameter(f"3D NFZ needs r1,c1,r2,c2,l — got '{s}'")
        return tuple(parts)
    else:
        if len(parts) != 4:
            raise typer.BadParameter(f"2D NFZ needs r1,c1,r2,c2 — got '{s}'")
        return tuple(parts)


def _default_connectivity(is_3d: bool) -> int:
    return 26 if is_3d else 8


def _default_heuristic(is_3d: bool) -> str:
    return "euclidean" if is_3d else "chebyshev"


def _build_grid(
    rows: int,
    cols: int,
    layers: int,
    obstacles: Optional[List[str]],
    nfzs: Optional[List[str]],
) -> tuple[Grid | Grid3D, bool]:
    is_3d = layers > 1
    if is_3d:
        grid: Grid | Grid3D = Grid3D(rows, cols, layers)
    else:
        grid = Grid(rows, cols)

    for obs in (obstacles or []):
        coord = _parse_coord(obs, is_3d)
        if is_3d:
            grid.add_obstacle(*coord)  # type: ignore[arg-type]
        else:
            grid.add_obstacle(*coord)  # type: ignore[arg-type]

    for nfz in (nfzs or []):
        coords = _parse_nfz(nfz, is_3d)
        if is_3d:
            grid.add_nfz(*coords)  # type: ignore[arg-type]
        else:
            grid.add_nfz(*coords)  # type: ignore[arg-type]

    return grid, is_3d


def _default_goal(rows: int, cols: int, layers: int, is_3d: bool) -> tuple:
    if is_3d:
        return (rows - 1, cols - 1, layers - 1)
    return (rows - 1, cols - 1)


def _print_stats(results: list[PathResult]) -> None:
    table = Table(title="Path Planning Results", show_header=True)
    table.add_column("Algorithm", style="cyan")
    table.add_column("Found", style="green")
    table.add_column("Path Length", justify="right")
    table.add_column("Nodes Explored", justify="right")
    table.add_column("Time (ms)", justify="right")
    for r in results:
        table.add_row(
            r.algorithm_name,
            "Yes" if r.found else "No",
            f"{r.path_length:.4f}",
            str(r.nodes_explored),
            f"{r.compute_time_ms:.4f}",
        )
    console.print(table)


@app.command()
def run(
    rows: int = typer.Option(20, help="Grid rows"),
    cols: int = typer.Option(20, help="Grid cols"),
    layers: int = typer.Option(1, help="Altitude layers (>1 = 3D mode)"),
    start: str = typer.Option("0,0", help="Start cell as r,c or r,c,l"),
    goal: Optional[str] = typer.Option(None, help="Goal cell (default: last cell)"),
    obstacle: Optional[List[str]] = typer.Option(None, help="Obstacle cell as r,c or r,c,l (repeatable)"),
    nfz: Optional[List[str]] = typer.Option(None, help="No-fly zone as r1,c1,r2,c2 or r1,c1,r2,c2,l (repeatable)"),
    algorithm: str = typer.Option("astar", help="Algorithm: astar or dijkstra"),
    heuristic: Optional[str] = typer.Option(None, help="Heuristic: manhattan, chebyshev, euclidean"),
    connectivity: Optional[int] = typer.Option(None, help="Connectivity: 4 or 8 (2D), 6 or 26 (3D)"),
    export: Optional[Path] = typer.Option(None, help="Export results to JSON"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Skip matplotlib display"),
) -> None:
    is_3d = layers > 1
    grid, _ = _build_grid(rows, cols, layers, obstacle, nfz)

    if start == "0,0" and is_3d:
        start_coord: tuple = (0, 0, 0)
    else:
        start_coord = _parse_coord(start, is_3d)
    goal_coord = _parse_coord(goal, is_3d) if goal else _default_goal(rows, cols, layers, is_3d)

    conn = connectivity if connectivity is not None else _default_connectivity(is_3d)
    heur = heuristic if heuristic is not None else _default_heuristic(is_3d)

    if algorithm == "astar":
        result = astar(grid, start_coord, goal_coord, connectivity=conn, heuristic=heur)
    else:
        result = dijkstra(grid, start_coord, goal_coord, connectivity=conn)

    _print_stats([result])

    if export:
        export_json(grid, [result], export)
        console.print(f"[green]Results exported to {export}[/green]")

    if not no_viz:
        if is_3d:
            plot_3d(grid, result, show=True)  # type: ignore[arg-type]
        else:
            plot_2d(grid, result, show=True)  # type: ignore[arg-type]


@app.command()
def compare(
    rows: int = typer.Option(20, help="Grid rows"),
    cols: int = typer.Option(20, help="Grid cols"),
    layers: int = typer.Option(1, help="Altitude layers (>1 = 3D mode)"),
    start: str = typer.Option("0,0", help="Start cell as r,c or r,c,l"),
    goal: Optional[str] = typer.Option(None, help="Goal cell (default: last cell)"),
    obstacle: Optional[List[str]] = typer.Option(None, help="Obstacle cell (repeatable)"),
    nfz: Optional[List[str]] = typer.Option(None, help="No-fly zone (repeatable)"),
    heuristic: Optional[str] = typer.Option(None, help="Heuristic for A*"),
    connectivity: Optional[int] = typer.Option(None, help="Connectivity"),
    export: Optional[Path] = typer.Option(None, help="Export results to JSON"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Skip matplotlib display"),
) -> None:
    is_3d = layers > 1
    grid, _ = _build_grid(rows, cols, layers, obstacle, nfz)

    if start == "0,0" and is_3d:
        start_coord: tuple = (0, 0, 0)
    else:
        start_coord = _parse_coord(start, is_3d)
    goal_coord = _parse_coord(goal, is_3d) if goal else _default_goal(rows, cols, layers, is_3d)
    conn = connectivity if connectivity is not None else _default_connectivity(is_3d)
    heur = heuristic if heuristic is not None else _default_heuristic(is_3d)

    d_result = dijkstra(grid, start_coord, goal_coord, connectivity=conn)
    a_result = astar(grid, start_coord, goal_coord, connectivity=conn, heuristic=heur)
    results = [a_result, d_result]

    _print_stats(results)

    if export:
        export_json(grid, results, export)
        console.print(f"[green]Results exported to {export}[/green]")

    if not no_viz:
        if is_3d:
            for r in results:
                plot_3d(grid, r, show=True)  # type: ignore[arg-type]
        else:
            plot_compare(grid, results, show=True)  # type: ignore[arg-type]
```

- [ ] **Step 4: Run `run` and `compare` CLI tests**

Run: `pytest tests/test_cli.py -v -k "run or compare"`
Expected: All PASS

---

## Task 11: CLI — `demo` Command

**Files:**
- Modify: `drone_planner/cli.py` (append `demo` command)
- Modify: `tests/test_cli.py` (append demo tests)

- [ ] **Step 1: Write failing demo tests**

Append to `tests/test_cli.py`:
```python
def test_demo_basic():
    result = runner.invoke(app, ["demo", "--scenario", "basic", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_demo_maze():
    result = runner.invoke(app, ["demo", "--scenario", "maze", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_demo_nfz_heavy():
    result = runner.invoke(app, ["demo", "--scenario", "nfz-heavy", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_demo_3d_layers():
    result = runner.invoke(app, ["demo", "--scenario", "3d-layers", "--no-viz"])
    assert result.exit_code == 0, result.output


def test_demo_invalid_scenario():
    result = runner.invoke(app, ["demo", "--scenario", "nonexistent", "--no-viz"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run to confirm demo tests fail**

Run: `pytest tests/test_cli.py -v -k "demo"`
Expected: Failures (demo command does not exist yet)

- [ ] **Step 3: Implement `demo` command — append to `drone_planner/cli.py`**

```python
_DEMO_SCENARIOS: dict[str, dict] = {
    "basic": {
        "description": "20x20 open grid with a small cluster of obstacles",
        "grid": {"rows": 20, "cols": 20, "layers": 1},
        "obstacles": [(5, 5), (5, 6), (5, 7), (6, 5), (6, 7), (7, 5), (7, 6), (7, 7)],
        "nfzs_2d": [],
        "start": (0, 0),
        "goal": (19, 19),
    },
    "maze": {
        "description": "20x20 grid with three long wall obstacles forcing a winding path",
        "grid": {"rows": 20, "cols": 20, "layers": 1},
        "walls": [
            # each wall: (r, c_start, c_end) or (r, c, r_end) — we encode as obstacle lists
        ],
        "obstacles": (
            [(5, c) for c in range(0, 15)]
            + [(10, c) for c in range(5, 20)]
            + [(15, c) for c in range(0, 15)]
        ),
        "nfzs_2d": [],
        "start": (0, 0),
        "goal": (19, 19),
    },
    "nfz-heavy": {
        "description": "20x20 grid with four large no-fly zones in quadrants",
        "grid": {"rows": 20, "cols": 20, "layers": 1},
        "obstacles": [],
        "nfzs_2d": [(2, 2, 7, 7), (2, 12, 7, 17), (12, 2, 17, 7), (12, 12, 17, 17)],
        "start": (0, 0),
        "goal": (19, 19),
    },
    "3d-layers": {
        "description": "10x10x4 grid with per-layer obstacles requiring multi-altitude routing",
        "grid": {"rows": 10, "cols": 10, "layers": 4},
        "obstacles_3d": [(4, 4, 1), (5, 5, 1), (4, 5, 2), (5, 4, 2)],
        "nfzs_3d": [(6, 6, 8, 8, 1)],
        "start": (0, 0, 0),
        "goal": (9, 9, 3),
    },
}


@app.command()
def demo(
    scenario: str = typer.Option("basic", help="Scenario: basic, maze, nfz-heavy, 3d-layers"),
    export: Optional[Path] = typer.Option(None, help="Export results to JSON"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Skip matplotlib display"),
) -> None:
    if scenario not in _DEMO_SCENARIOS:
        valid = ", ".join(_DEMO_SCENARIOS.keys())
        console.print(f"[red]Unknown scenario '{scenario}'. Valid: {valid}[/red]")
        raise typer.Exit(code=1)

    cfg = _DEMO_SCENARIOS[scenario]
    console.print(f"[cyan]Demo: {scenario}[/cyan] — {cfg['description']}")

    is_3d = cfg["grid"]["layers"] > 1
    rows, cols, layers = cfg["grid"]["rows"], cfg["grid"]["cols"], cfg["grid"]["layers"]

    if is_3d:
        grid: Grid | Grid3D = Grid3D(rows, cols, layers)
        for obs in cfg.get("obstacles_3d", []):
            grid.add_obstacle(*obs)  # type: ignore[arg-type]
        for nfz_coords in cfg.get("nfzs_3d", []):
            grid.add_nfz(*nfz_coords)  # type: ignore[arg-type]
    else:
        grid = Grid(rows, cols)
        for obs in cfg.get("obstacles", []):
            grid.add_obstacle(*obs)  # type: ignore[arg-type]
        for nfz_coords in cfg.get("nfzs_2d", []):
            grid.add_nfz(*nfz_coords)  # type: ignore[arg-type]

    start_coord = cfg["start"]
    goal_coord = cfg["goal"]
    conn = 26 if is_3d else 8
    heur = "euclidean" if is_3d else "chebyshev"

    d_result = dijkstra(grid, start_coord, goal_coord, connectivity=conn)
    a_result = astar(grid, start_coord, goal_coord, connectivity=conn, heuristic=heur)
    results = [a_result, d_result]

    _print_stats(results)

    if export:
        export_json(grid, results, export)
        console.print(f"[green]Results exported to {export}[/green]")

    if not no_viz:
        if is_3d:
            for r in results:
                plot_3d(grid, r, show=True)  # type: ignore[arg-type]
        else:
            plot_compare(grid, results, show=True)  # type: ignore[arg-type]
```

- [ ] **Step 4: Run all CLI tests**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

---

## Task 12: Full Test Suite Pass + README

**Files:**
- Modify: `README.md` (full overwrite)

- [ ] **Step 1: Run the full test suite and confirm all pass**

Run: `pytest -v`
Expected: All tests PASS, zero failures

If any tests fail, fix the underlying issue before continuing.

- [ ] **Step 2: Write README.md**

```markdown
# Drone Path Planner

Autonomous drone path planning simulation implementing A* and Dijkstra across configurable 2D and 3D grid environments with obstacle avoidance, no-fly zones, matplotlib visualization, and JSON export.

## Features

- **A\* and Dijkstra** pathfinding with configurable heuristics
- **2D and 3D** grid environments (altitude layers)
- **Obstacles** (individual cells) and **No-Fly Zones** (rectangular regions, rendered distinctly)
- **Side-by-side comparison** of both algorithms with stats table
- **4/8-connected** movement in 2D; **6/26-connected** in 3D (configurable)
- **matplotlib visualization** — 2D color-coded grid and interactive 3D voxel plot
- **JSON export** for downstream analysis
- **Demo scenarios** — prebuilt grids for quick showcase

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run a single algorithm

```bash
python -m drone_planner run --rows 20 --cols 20 --algorithm astar
python -m drone_planner run --rows 20 --cols 20 --algorithm dijkstra --connectivity 4
```

With obstacles and a no-fly zone:

```bash
python -m drone_planner run \
  --obstacle 5,5 --obstacle 5,6 --obstacle 6,5 \
  --nfz 10,10,14,14 \
  --algorithm astar \
  --export results.json
```

### Compare A* vs Dijkstra side-by-side

```bash
python -m drone_planner compare --rows 20 --cols 20
python -m drone_planner compare --nfz 5,5,10,10 --export comparison.json
```

Example terminal output:

```
          Path Planning Results
┌───────────┬───────┬─────────────┬────────────────┬───────────┐
│ Algorithm │ Found │ Path Length │ Nodes Explored │ Time (ms) │
├───────────┼───────┼─────────────┼────────────────┼───────────┤
│ astar     │ Yes   │ 26.8701     │ 142            │ 0.8210    │
│ dijkstra  │ Yes   │ 26.8701     │ 381            │ 1.4390    │
└───────────┴───────┴─────────────┴────────────────┴───────────┘
```

### 3D mode (altitude layers)

```bash
python -m drone_planner run \
  --rows 10 --cols 10 --layers 4 \
  --start 0,0,0 --goal 9,9,3 \
  --obstacle 4,4,1 --nfz 6,6,8,8,2 \
  --algorithm astar
```

### Demo scenarios

```bash
python -m drone_planner demo --scenario basic
python -m drone_planner demo --scenario maze
python -m drone_planner demo --scenario nfz-heavy
python -m drone_planner demo --scenario 3d-layers
```

Available scenarios:

| Scenario | Description |
|---|---|
| `basic` | 20×20 grid, small obstacle cluster |
| `maze` | 20×20 grid, three long walls |
| `nfz-heavy` | 20×20 grid, four large no-fly zones in quadrants |
| `3d-layers` | 10×10×4 grid, per-altitude obstacles |

### Export to JSON

```bash
python -m drone_planner compare --export results.json --no-viz
```

Output format:

```json
{
  "grid": { "rows": 20, "cols": 20, "layers": 1 },
  "algorithms": [
    {
      "name": "astar",
      "path": [[0,0],[1,1],"..."],
      "path_length": 26.8701,
      "nodes_explored": 142,
      "compute_time_ms": 0.821,
      "found": true
    }
  ]
}
```

## Visualization

**2D plot:**
- White = walkable, dark gray = obstacle, red with `////` = no-fly zone
- Light blue = explored nodes, green line = path, ★ = start/goal

**3D plot:**
- Gray voxels = obstacles, red voxels = no-fly zones
- Green line = path through 3D voxel space, interactive rotation

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--rows` | 20 | Grid rows |
| `--cols` | 20 | Grid cols |
| `--layers` | 1 | Altitude layers (>1 enables 3D) |
| `--start` | `0,0` | Start cell (`r,c` or `r,c,l`) |
| `--goal` | last cell | Goal cell (`r,c` or `r,c,l`) |
| `--obstacle` | — | Repeatable; `r,c` or `r,c,l` |
| `--nfz` | — | Repeatable; `r1,c1,r2,c2` or `r1,c1,r2,c2,l` |
| `--algorithm` | `astar` | `astar` or `dijkstra` |
| `--heuristic` | `chebyshev`/`euclidean` | `manhattan`, `chebyshev`, `euclidean` |
| `--connectivity` | `8`/`26` | `4` or `8` (2D); `6` or `26` (3D) |
| `--export` | — | Write JSON to path |
| `--no-viz` | off | Skip matplotlib display |

## Running Tests

```bash
pytest -v
```

## Dependencies

- `typer[all]` — CLI framework
- `matplotlib` — 2D and 3D visualization
- `numpy` — grid storage
- `rich` — terminal tables
- `pytest` — test suite
```

- [ ] **Step 3: Run the full test suite one final time**

Run: `pytest -v`
Expected: All tests PASS, zero failures
```
