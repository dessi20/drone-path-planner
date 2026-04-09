# Drone Path Planner — Design Spec
Date: 2026-04-09

## Overview

A Python CLI tool that simulates autonomous drone path planning using A* and Dijkstra algorithms across configurable 2D and 3D grid environments. Supports obstacle avoidance, no-fly zones, side-by-side algorithm comparison, matplotlib visualization, and JSON export.

---

## Architecture

**Package layout:**

```
drone-path-planner/
├── drone_planner/
│   ├── __init__.py
│   ├── grid.py        # Grid, Grid3D, cell state, obstacles, NoFlyZone
│   ├── algorithms.py  # AStar, Dijkstra, PathResult dataclass
│   ├── visualizer.py  # 2D matplotlib plots + 3D voxel plots
│   ├── cli.py         # Typer app (entry point)
│   └── exporter.py    # JSON serialization of results
├── tests/
│   ├── test_grid.py
│   ├── test_algorithms.py
│   ├── test_exporter.py
│   └── test_visualizer.py
├── requirements.txt
└── README.md
```

**Data flow:**
CLI parses args → builds `Grid` or `Grid3D` → runs one or both algorithms → collects `PathResult` dataclasses → passes to `Visualizer` for plotting and/or `Exporter` for JSON output.

---

## Module Design

### `grid.py`

Owns grid dimensions, cell walkability, obstacle placement, and no-fly zone registration.

- **Cell states:** `EMPTY`, `OBSTACLE`, `NO_FLY_ZONE`
- **`Grid(rows, cols)`** — 2D grid. Methods:
  - `add_obstacle(r, c)` — mark individual cell as impassable
  - `add_nfz(r1, c1, r2, c2)` — register rectangular no-fly zone; marks all cells in rectangle as `NO_FLY_ZONE`
  - `is_walkable(r, c)` — returns True only for `EMPTY` cells
  - `neighbors(r, c, connectivity=8)` — returns valid neighbor coords for 4 or 8-connected movement
- **`Grid3D(rows, cols, layers)`** — subclasses Grid, adds altitude dimension:
  - `add_obstacle(r, c, layer)`
  - `add_nfz(r1, c1, r2, c2, layer)`
  - `neighbors(r, c, layer, connectivity=26)` — 6 or 26-connected movement

No-fly zones and obstacles are both hard-blocked (impassable). They differ only in cell state type (for rendering) and how they are defined (individual cell vs rectangle).

### `algorithms.py`

Implements A* and Dijkstra. Both algorithms accept a grid, start, goal, and connectivity setting, and return a `PathResult`.

```python
@dataclass
class PathResult:
    algorithm_name: str
    path: list[tuple]      # List of (r, c) or (r, c, layer) coords
    path_length: float     # Sum of edge weights along path
    nodes_explored: int
    compute_time_ms: float
    found: bool            # False if no path exists
```

**A\*:**
- Heuristic selectable: `manhattan` (4-conn), `chebyshev` (8-conn default), `euclidean` (3D default)
- Uses `heapq` priority queue on `f = g + h`
- Diagonal moves cost √2, vertical moves cost 1, combined diagonal+vertical moves cost √3

**Dijkstra:**
- Uniform cost search; same edge weight scheme as A*
- No heuristic; guaranteed optimal

**Connectivity:**
- 2D: 4-connected (cardinal) or 8-connected (cardinal + diagonal, default)
- 3D: 6-connected (face-adjacent) or 26-connected (all neighbors including edge/corner, default)

### `visualizer.py`

**2D visualization** (single algorithm):

| Element | Rendering |
|---|---|
| Walkable cells | White |
| Obstacle cells | Dark gray |
| No-fly zones | Red with `////` crosshatch |
| Explored nodes | Light blue fill |
| Final path | Green line + dots |
| Start | Green star marker |
| Goal | Red star marker |

**Compare visualization**: Two subplots side-by-side (A* left, Dijkstra right) with identical grid rendering. Stats table shown below using `rich` in the terminal.

**3D visualization**: `mpl_toolkits.mplot3d` with voxel rendering. Obstacles/NFZs as colored voxel blocks at their altitudes. Path as a 3D line through voxel centers. Start/goal as 3D scatter markers. Default camera: 30° elevation, 45° azimuth (interactive window, user can rotate).

### `cli.py`

Entry point using Typer. Three commands:

**`run`** — single algorithm on a configured grid:
```
--rows INT              Grid rows (default: 20)
--cols INT              Grid cols (default: 20)
--layers INT            Altitude layers; triggers 3D mode (default: 1 = 2D)
--start "r,c[,l]"       Start cell (default: "0,0" or "0,0,0")
--goal  "r,c[,l]"       Goal cell (default: last cell)
--obstacle "r,c[,l]"    Repeatable; individual obstacle cell
--nfz "r1,c1,r2,c2[,l]" Repeatable; no-fly zone rectangle
--algorithm [astar|dijkstra]  (default: astar)
--heuristic [manhattan|chebyshev|euclidean]  (default: chebyshev in 2D, euclidean in 3D)
--connectivity [4|8]  (2D mode) or [6|26] (3D mode); default 8 in 2D, 26 in 3D; CLI validates mode/value compatibility
--export PATH           Write JSON result to file
--no-viz                Skip matplotlib display
```

**`compare`** — same grid flags as `run`, runs both A* and Dijkstra, displays side-by-side figure and terminal stats table.

**`demo`** — named preset scenarios, no grid setup required:
```
--scenario [basic|maze|nfz-heavy|3d-layers]
```

### `exporter.py`

Serializes results to JSON:

```json
{
  "grid": {
    "rows": 20,
    "cols": 20,
    "layers": 1
  },
  "algorithms": [
    {
      "name": "astar",
      "path": [[0, 0], [0, 1], "..."],
      "path_length": 32.4,
      "nodes_explored": 118,
      "compute_time_ms": 1.4,
      "found": true
    }
  ]
}
```

---

## Testing

**Framework:** pytest, no mocks on core logic — algorithms run against real in-memory grids. Visualizer tests use matplotlib `Agg` backend.

**`test_grid.py`:**
- Grid construction with correct dimensions
- `add_obstacle` marks correct cell
- `add_nfz` marks all cells in rectangle
- Boundary cells handled correctly
- 3D layer indexing works

**`test_algorithms.py`:**
- Both algorithms find a valid path on open grid
- Both return `found=False` when goal is unreachable
- Path starts at `start`, ends at `goal`
- A* path length ≤ Dijkstra path length + tolerance (admissible heuristic)
- 4-connected vs 8-connected produce different but valid paths
- 3D pathfinding navigates between altitude layers
- `nodes_explored` and `compute_time_ms` populated in result

**`test_exporter.py`:**
- JSON output is valid and contains expected top-level keys
- Path coordinates round-trip correctly
- File written to specified path

**`test_visualizer.py`:**
- 2D run figure has 1 axes
- Compare figure has 2 axes
- 3D figure has 1 `Axes3D` axes
- No errors raised during figure creation

---

## Dependencies (`requirements.txt`)

```
typer[all]
matplotlib
numpy
rich
pytest
```

---

## Non-Goals

- No real drone hardware integration
- No dynamic re-planning (obstacles are static per run)
- No plugin architecture or ABC hierarchy for algorithms
- No backwards-compatibility shims
