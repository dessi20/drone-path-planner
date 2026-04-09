from __future__ import annotations

import matplotlib
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed to register 3D projection

from drone_planner.grid import CellState, Grid, Grid3D
from drone_planner.algorithms import PathResult


def _render_grid_on_ax(ax, grid: Grid, result: PathResult) -> None:
    """Render the grid cells, explored nodes, and path onto a given Axes."""
    rows, cols = grid.rows, grid.cols

    explored_set = set(result.explored_nodes)

    for r in range(rows):
        for c in range(cols):
            state = grid.cell_state(r, c)
            x, y = c, r  # bottom-left corner of cell

            if state == CellState.OBSTACLE:
                patch = mpatches.Rectangle(
                    (x, y), 1, 1,
                    facecolor='#444444',
                    edgecolor='black',
                    linewidth=0.5,
                )
                ax.add_patch(patch)
            elif state == CellState.NO_FLY_ZONE:
                patch = mpatches.Rectangle(
                    (x, y), 1, 1,
                    facecolor='#cc0000',
                    hatch='////',
                    edgecolor='#880000',
                    linewidth=0.5,
                )
                ax.add_patch(patch)
            else:
                # EMPTY cell — check if explored
                if (r, c) in explored_set:
                    patch = mpatches.Rectangle(
                        (x, y), 1, 1,
                        facecolor='#add8e6',
                        edgecolor='#aaaaaa',
                        linewidth=0.3,
                    )
                    ax.add_patch(patch)
                # else: white background (default axes background)

    # Draw path
    if result.path:
        path_x = [c + 0.5 for r, c in result.path]
        path_y = [r + 0.5 for r, c in result.path]

        ax.plot(path_x, path_y, color='green', linewidth=2, zorder=4)
        ax.scatter(path_x, path_y, color='green', s=20, zorder=5)

        # Start marker (green star)
        start_x = result.path[0][1] + 0.5
        start_y = result.path[0][0] + 0.5
        ax.plot(start_x, start_y, 'g*', markersize=15, zorder=6)

        # Goal marker (red star)
        goal_x = result.path[-1][1] + 0.5
        goal_y = result.path[-1][0] + 0.5
        ax.plot(goal_x, goal_y, 'r*', markersize=15, zorder=6)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(result.algorithm_name)


def plot_2d(grid: Grid, result: PathResult, show: bool = True) -> matplotlib.figure.Figure:
    """Render a 2D grid with a single PathResult.

    Parameters
    ----------
    grid : Grid
        The grid to render.
    result : PathResult
        The path-finding result containing the path and explored nodes.
    show : bool
        If True, call plt.show() after rendering.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    _render_grid_on_ax(ax, grid, result)

    if show:
        plt.show()

    return fig


def plot_compare(
    grid: Grid,
    results: list[PathResult],
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Render two PathResults side-by-side for comparison.

    Parameters
    ----------
    grid : Grid
        The grid to render.
    results : list[PathResult]
        A list of exactly two PathResult objects (e.g. A* and Dijkstra).
    show : bool
        If True, call plt.show() after rendering.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, result in zip(axes, results):
        _render_grid_on_ax(ax, grid, result)

    fig.suptitle("Algorithm Comparison")

    if show:
        plt.show()

    return fig


def plot_3d(grid3d: Grid3D, result: PathResult, show: bool = True) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Render obstacle and NFZ voxels as colored blocks
    # For each (r, c, layer) that is OBSTACLE or NO_FLY_ZONE, draw a small cube
    # Use ax.bar3d(x, y, z, dx, dy, dz, color, alpha) where each cell at (r,c,layer)
    # maps to bar3d(c, r, layer, 1, 1, 1, color, alpha)
    for layer in range(grid3d.layers):
        for r in range(grid3d.rows):
            for c in range(grid3d.cols):
                state = grid3d.cell_state(r, c, layer)
                if state == CellState.OBSTACLE:
                    ax.bar3d(c, r, layer, 1, 1, 1, color='#444444', alpha=0.7)
                elif state == CellState.NO_FLY_ZONE:
                    ax.bar3d(c, r, layer, 1, 1, 1, color='#cc0000', alpha=0.7)

    # Draw path as 3D line through cell centers
    # For 3D path, tuples are (r, c, layer)
    if result.path:
        px = [c + 0.5 for r, c, layer in result.path]
        py = [r + 0.5 for r, c, layer in result.path]
        pz = [layer + 0.5 for r, c, layer in result.path]
        ax.plot(px, py, pz, color='green', linewidth=2)

        # Start marker
        ax.scatter([px[0]], [py[0]], [pz[0]], color='green', marker='*', s=200)
        # Goal marker
        ax.scatter([px[-1]], [py[-1]], [pz[-1]], color='red', marker='*', s=200)

    ax.set_xlabel('Col')
    ax.set_ylabel('Row')
    ax.set_zlabel('Layer')
    ax.set_title(f"3D Path \u2014 {result.algorithm_name}")
    ax.view_init(elev=30, azim=45)

    if show:
        plt.show()

    return fig
