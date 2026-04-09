import matplotlib
matplotlib.use('Agg')

import matplotlib.figure
import pytest

from drone_planner.grid import Grid
from drone_planner.algorithms import dijkstra, astar
from drone_planner.visualizer import plot_2d, plot_compare


def test_plot_2d_returns_figure():
    grid = Grid(10, 10)
    result = dijkstra(grid, (0, 0), (9, 9))
    fig = plot_2d(grid, result, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_2d_has_one_axes():
    grid = Grid(10, 10)
    result = dijkstra(grid, (0, 0), (9, 9))
    fig = plot_2d(grid, result, show=False)
    assert len(fig.axes) == 1


def test_plot_2d_with_obstacles():
    grid = Grid(10, 10)
    grid.add_obstacle(2, 3)
    grid.add_obstacle(4, 5)
    grid.add_obstacle(7, 2)
    grid.add_nfz(1, 7, 3, 9)
    result = dijkstra(grid, (0, 0), (9, 9))
    fig = plot_2d(grid, result, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_2d_no_path():
    # Block the entire column 5 to make (0,0)->(9,9) unreachable
    # by walling off a continuous barrier the path cannot cross
    grid = Grid(10, 10)
    # Block column 1 fully so goal (9,9) is unreachable from (0,0)
    for r in range(10):
        grid.add_obstacle(r, 1)
    result = dijkstra(grid, (0, 0), (9, 9))
    assert result.found is False
    assert result.path == []
    # Must handle empty path gracefully
    fig = plot_2d(grid, result, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_compare_returns_figure():
    grid = Grid(10, 10)
    r1 = dijkstra(grid, (0, 0), (9, 9))
    r2 = astar(grid, (0, 0), (9, 9))
    fig = plot_compare(grid, [r1, r2], show=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_compare_has_two_axes():
    grid = Grid(10, 10)
    r1 = dijkstra(grid, (0, 0), (9, 9))
    r2 = astar(grid, (0, 0), (9, 9))
    fig = plot_compare(grid, [r1, r2], show=False)
    assert len(fig.axes) == 2


def test_plot_3d_returns_figure():
    from drone_planner.grid import Grid3D
    from drone_planner.algorithms import dijkstra
    from drone_planner.visualizer import plot_3d
    g = Grid3D(5, 5, 3)
    result = dijkstra(g, (0, 0, 0), (4, 4, 2))
    fig = plot_3d(g, result, show=False)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_3d_has_axes3d():
    from drone_planner.grid import Grid3D
    from drone_planner.algorithms import dijkstra
    from drone_planner.visualizer import plot_3d
    from mpl_toolkits.mplot3d import Axes3D
    g = Grid3D(5, 5, 3)
    result = dijkstra(g, (0, 0, 0), (4, 4, 2))
    fig = plot_3d(g, result, show=False)
    assert len(fig.axes) == 1
    assert isinstance(fig.axes[0], Axes3D)
