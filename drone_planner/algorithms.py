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


def dijkstra(
    grid: Union[Grid, Grid3D],
    start: tuple,
    goal: tuple,
    connectivity: int | None = None,
) -> PathResult:
    start_time = time.perf_counter()
    is_3d = isinstance(grid, Grid3D)
    if connectivity is None:
        connectivity = 26 if is_3d else 8

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
    connectivity: int | None = None,
    heuristic: str = "chebyshev",
) -> PathResult:
    start_time = time.perf_counter()
    is_3d = isinstance(grid, Grid3D)
    if connectivity is None:
        connectivity = 26 if is_3d else 8

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
