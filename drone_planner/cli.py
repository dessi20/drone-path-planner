from __future__ import annotations

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table

from drone_planner.grid import Grid, Grid3D
from drone_planner.algorithms import astar, dijkstra
from drone_planner.visualizer import plot_2d, plot_compare, plot_3d
from drone_planner.exporter import export_json

app = typer.Typer()
console = Console()


def _parse_coord(s: str, is_3d: bool) -> tuple:
    parts = s.split(",")
    expected = 3 if is_3d else 2
    if len(parts) != expected:
        raise typer.BadParameter(
            f"Expected {'r,c,l' if is_3d else 'r,c'} but got {s!r}"
        )
    return tuple(int(p) for p in parts)


def _build_grid(
    rows: int,
    cols: int,
    layers: int,
    obstacles: list[str],
    nfzs: list[str],
) -> tuple:
    is_3d = layers > 1
    grid = Grid3D(rows, cols, layers) if is_3d else Grid(rows, cols)

    for obs in obstacles:
        parts = obs.split(",")
        if is_3d:
            r, c, l = int(parts[0]), int(parts[1]), int(parts[2])
            grid.add_obstacle(r, c, l)
        else:
            r, c = int(parts[0]), int(parts[1])
            grid.add_obstacle(r, c)

    for nfz in nfzs:
        parts = nfz.split(",")
        if is_3d:
            r1, c1, r2, c2, l = (
                int(parts[0]), int(parts[1]),
                int(parts[2]), int(parts[3]),
                int(parts[4]),
            )
            grid.add_nfz(r1, c1, r2, c2, l)
        else:
            r1, c1, r2, c2 = (
                int(parts[0]), int(parts[1]),
                int(parts[2]), int(parts[3]),
            )
            grid.add_nfz(r1, c1, r2, c2)

    return grid, is_3d


@app.command()
def run(
    rows: int = typer.Option(20, help="Grid rows"),
    cols: int = typer.Option(20, help="Grid cols"),
    layers: int = typer.Option(1, help="Altitude layers; >1 enables 3D mode"),
    start: str = typer.Option("0,0", help='Start cell, e.g. "0,0" or "0,0,0"'),
    goal: str = typer.Option("", help="Goal cell; defaults to last cell"),
    obstacle: Optional[List[str]] = typer.Option(None, help="Obstacle cell (repeatable)"),
    nfz: Optional[List[str]] = typer.Option(None, help="No-fly zone rectangle (repeatable)"),
    algorithm: str = typer.Option("astar", help="Algorithm: astar or dijkstra"),
    heuristic: str = typer.Option("", help="Heuristic (A* only): manhattan, chebyshev, euclidean"),
    connectivity: int = typer.Option(0, help="Connectivity: 4/8 (2D) or 6/26 (3D); 0=default"),
    export: str = typer.Option("", help="Write JSON result to this file path"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Skip matplotlib display"),
) -> None:
    is_3d = layers > 1

    start_coord = _parse_coord(start, is_3d)

    if goal == "":
        goal_coord = (rows - 1, cols - 1, layers - 1) if is_3d else (rows - 1, cols - 1)
    else:
        goal_coord = _parse_coord(goal, is_3d)

    grid, is_3d = _build_grid(rows, cols, layers, obstacle or [], nfz or [])

    conn = connectivity if connectivity != 0 else (26 if is_3d else 8)

    if heuristic == "":
        heuristic_val = "euclidean" if is_3d else "chebyshev"
    else:
        heuristic_val = heuristic

    if algorithm == "dijkstra":
        result = dijkstra(grid, start_coord, goal_coord, connectivity=conn)
    else:
        result = astar(grid, start_coord, goal_coord, connectivity=conn, heuristic=heuristic_val)

    if result.found:
        console.print(f"[green]Path found![/green]")
        console.print(f"  Algorithm:      {result.algorithm_name}")
        console.print(f"  Path length:    {result.path_length:.4f}")
        console.print(f"  Nodes explored: {result.nodes_explored}")
        console.print(f"  Time (ms):      {result.compute_time_ms:.4f}")
    else:
        console.print(f"[red]No path found.[/red]")
        console.print(f"  Algorithm:      {result.algorithm_name}")
        console.print(f"  Nodes explored: {result.nodes_explored}")
        console.print(f"  Time (ms):      {result.compute_time_ms:.4f}")

    if export:
        export_json(grid, [result], export)
        console.print(f"Result exported to {export}")

    if not no_viz:
        if is_3d:
            plot_3d(grid, result)
        else:
            plot_2d(grid, result)


@app.command()
def compare(
    rows: int = typer.Option(20, help="Grid rows"),
    cols: int = typer.Option(20, help="Grid cols"),
    layers: int = typer.Option(1, help="Altitude layers; >1 enables 3D mode"),
    start: str = typer.Option("0,0", help='Start cell, e.g. "0,0" or "0,0,0"'),
    goal: str = typer.Option("", help="Goal cell; defaults to last cell"),
    obstacle: Optional[List[str]] = typer.Option(None, help="Obstacle cell (repeatable)"),
    nfz: Optional[List[str]] = typer.Option(None, help="No-fly zone rectangle (repeatable)"),
    heuristic: str = typer.Option("", help="Heuristic for A*: manhattan, chebyshev, euclidean"),
    connectivity: int = typer.Option(0, help="Connectivity: 4/8 (2D) or 6/26 (3D); 0=default"),
    export: str = typer.Option("", help="Write JSON result to this file path"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Skip matplotlib display"),
) -> None:
    is_3d = layers > 1

    start_coord = _parse_coord(start, is_3d)

    if goal == "":
        goal_coord = (rows - 1, cols - 1, layers - 1) if is_3d else (rows - 1, cols - 1)
    else:
        goal_coord = _parse_coord(goal, is_3d)

    grid, is_3d = _build_grid(rows, cols, layers, obstacle or [], nfz or [])

    conn = connectivity if connectivity != 0 else (26 if is_3d else 8)

    if heuristic == "":
        heuristic_val = "euclidean" if is_3d else "chebyshev"
    else:
        heuristic_val = heuristic

    r1 = astar(grid, start_coord, goal_coord, connectivity=conn, heuristic=heuristic_val)
    r2 = dijkstra(grid, start_coord, goal_coord, connectivity=conn)

    table = Table(title="Algorithm Comparison")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Found", style="green")
    table.add_column("Path Length", justify="right")
    table.add_column("Nodes Explored", justify="right")
    table.add_column("Time (ms)", justify="right")

    for res in (r1, r2):
        table.add_row(
            res.algorithm_name,
            str(res.found),
            f"{res.path_length:.4f}",
            str(res.nodes_explored),
            f"{res.compute_time_ms:.4f}",
        )

    console.print(table)

    if export:
        export_json(grid, [r1, r2], export)
        console.print(f"Results exported to {export}")

    if not no_viz:
        plot_compare(grid, [r1, r2])


def _print_single_result(result) -> None:
    status = "[green]Found[/green]" if result.found else "[red]Not found[/red]"
    console.print(f"  {result.algorithm_name}: {status} | length={result.path_length:.4f} | nodes={result.nodes_explored} | time={result.compute_time_ms:.4f}ms")


def _print_compare_table(results) -> None:
    table = Table(title="Comparison")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Found")
    table.add_column("Path Length", justify="right")
    table.add_column("Nodes Explored", justify="right")
    table.add_column("Time (ms)", justify="right")
    for res in results:
        table.add_row(
            res.algorithm_name,
            "[green]Yes[/green]" if res.found else "[red]No[/red]",
            f"{res.path_length:.4f}",
            str(res.nodes_explored),
            f"{res.compute_time_ms:.4f}",
        )
    console.print(table)


def _demo_basic(no_viz: bool, export: str) -> None:
    console.print("[bold]Demo: Basic[/bold] — 20×20 grid, A* vs Dijkstra")
    grid = Grid(20, 20)
    for r, c in [(5, 3), (5, 4), (5, 5), (10, 8), (10, 9), (15, 12)]:
        grid.add_obstacle(r, c)
    r1 = astar(grid, (0, 0), (19, 19))
    r2 = dijkstra(grid, (0, 0), (19, 19))
    _print_compare_table([r1, r2])
    if export:
        export_json(grid, [r1, r2], export)
    if not no_viz:
        plot_compare(grid, [r1, r2])


def _demo_maze(no_viz: bool, export: str) -> None:
    console.print("[bold]Demo: Maze[/bold] — 20×20 grid with wall barriers")
    grid = Grid(20, 20)
    # Wall 1: row 5, cols 0-14 (gap at 15-19)
    for c in range(15):
        grid.add_obstacle(5, c)
    # Wall 2: row 10, cols 5-19 (gap at 0-4)
    for c in range(5, 20):
        grid.add_obstacle(10, c)
    # Wall 3: row 15, cols 0-14 (gap at 15-19)
    for c in range(15):
        grid.add_obstacle(15, c)
    r1 = astar(grid, (0, 0), (19, 19))
    r2 = dijkstra(grid, (0, 0), (19, 19))
    _print_compare_table([r1, r2])
    if export:
        export_json(grid, [r1, r2], export)
    if not no_viz:
        plot_compare(grid, [r1, r2])


def _demo_nfz_heavy(no_viz: bool, export: str) -> None:
    console.print("[bold]Demo: NFZ-Heavy[/bold] — 20×20 grid with large no-fly zones")
    grid = Grid(20, 20)
    grid.add_nfz(3, 3, 7, 7)
    grid.add_nfz(3, 12, 7, 17)
    grid.add_nfz(12, 5, 16, 14)
    result = astar(grid, (0, 0), (19, 19))
    _print_single_result(result)
    if export:
        export_json(grid, [result], export)
    if not no_viz:
        plot_2d(grid, result)


def _demo_3d_layers(no_viz: bool, export: str) -> None:
    console.print("[bold]Demo: 3D Layers[/bold] — 10×10×4 grid, multi-layer navigation")
    grid = Grid3D(10, 10, 4)
    grid.add_obstacle(3, 3, 0)
    grid.add_obstacle(3, 4, 0)
    grid.add_obstacle(5, 5, 1)
    grid.add_obstacle(5, 6, 1)
    grid.add_obstacle(7, 2, 2)
    grid.add_obstacle(7, 3, 2)
    grid.add_nfz(2, 2, 4, 4, 3)
    r1 = astar(grid, (0, 0, 0), (9, 9, 3))
    r2 = dijkstra(grid, (0, 0, 0), (9, 9, 3))
    _print_compare_table([r1, r2])
    if export:
        export_json(grid, [r1, r2], export)
    if not no_viz:
        plot_3d(grid, r1)


@app.command()
def demo(
    scenario: str = typer.Option("basic", help="Scenario: basic, maze, nfz-heavy, 3d-layers"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Skip matplotlib display"),
    export: str = typer.Option("", help="Write JSON result to this file path"),
) -> None:
    """Run a named demo scenario."""
    scenarios = {
        "basic": _demo_basic,
        "maze": _demo_maze,
        "nfz-heavy": _demo_nfz_heavy,
        "3d-layers": _demo_3d_layers,
    }
    if scenario not in scenarios:
        console.print(f"[red]Unknown scenario: {scenario!r}. Choose from: {', '.join(scenarios)}[/red]")
        raise typer.Exit(code=1)
    scenarios[scenario](no_viz=no_viz, export=export)
