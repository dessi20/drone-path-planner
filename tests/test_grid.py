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


def test_add_nfz_reversed_corners():
    """add_nfz should work regardless of corner order."""
    g = Grid(10, 10)
    g.add_nfz(3, 3, 1, 1)  # reversed from (1,1,3,3)
    for r in range(1, 4):
        for c in range(1, 4):
            assert g.cell_state(r, c) == CellState.NO_FLY_ZONE


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
