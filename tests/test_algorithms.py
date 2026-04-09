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
