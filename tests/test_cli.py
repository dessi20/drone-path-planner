from __future__ import annotations

import pytest
from typer.testing import CliRunner

from drone_planner.cli import app

runner = CliRunner()


def test_run_basic_2d():
    result = runner.invoke(app, ["run", "--no-viz"])
    assert result.exit_code == 0


def test_run_astar():
    result = runner.invoke(app, ["run", "--algorithm", "astar", "--no-viz"])
    assert result.exit_code == 0


def test_run_dijkstra():
    result = runner.invoke(app, ["run", "--algorithm", "dijkstra", "--no-viz"])
    assert result.exit_code == 0


def test_run_with_obstacles():
    result = runner.invoke(
        app,
        [
            "run",
            "--rows", "10",
            "--cols", "10",
            "--obstacle", "3,3",
            "--obstacle", "3,4",
            "--no-viz",
        ],
    )
    assert result.exit_code == 0


def test_run_with_nfz():
    result = runner.invoke(
        app,
        [
            "run",
            "--rows", "10",
            "--cols", "10",
            "--nfz", "2,2,4,4",
            "--no-viz",
        ],
    )
    assert result.exit_code == 0


def test_run_no_path():
    obstacles = []
    for r in range(10):
        obstacles += ["--obstacle", f"{r},5"]
    result = runner.invoke(
        app,
        [
            "run",
            "--rows", "10",
            "--cols", "10",
            "--start", "0,0",
            "--goal", "0,9",
            "--algorithm", "dijkstra",
            "--no-viz",
        ]
        + obstacles,
    )
    assert result.exit_code == 0


def test_run_export(tmp_path):
    out_file = tmp_path / "out.json"
    result = runner.invoke(app, ["run", "--no-viz", "--export", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()


def test_compare_basic():
    result = runner.invoke(app, ["compare", "--no-viz"])
    assert result.exit_code == 0


def test_compare_with_grid_flags():
    result = runner.invoke(
        app,
        [
            "compare",
            "--rows", "15",
            "--cols", "15",
            "--connectivity", "4",
            "--no-viz",
        ],
    )
    assert result.exit_code == 0


def test_run_3d():
    result = runner.invoke(
        app,
        [
            "run",
            "--rows", "5",
            "--cols", "5",
            "--layers", "3",
            "--start", "0,0,0",
            "--goal", "4,4,2",
            "--no-viz",
        ],
    )
    assert result.exit_code == 0


def test_demo_basic():
    result = runner.invoke(app, ["demo", "--scenario", "basic", "--no-viz"])
    assert result.exit_code == 0

def test_demo_maze():
    result = runner.invoke(app, ["demo", "--scenario", "maze", "--no-viz"])
    assert result.exit_code == 0

def test_demo_nfz_heavy():
    result = runner.invoke(app, ["demo", "--scenario", "nfz-heavy", "--no-viz"])
    assert result.exit_code == 0

def test_demo_3d_layers():
    result = runner.invoke(app, ["demo", "--scenario", "3d-layers", "--no-viz"])
    assert result.exit_code == 0

def test_demo_invalid_scenario():
    result = runner.invoke(app, ["demo", "--scenario", "nonexistent", "--no-viz"])
    assert result.exit_code == 1

def test_demo_export(tmp_path):
    result = runner.invoke(app, ["demo", "--scenario", "basic", "--no-viz", "--export", str(tmp_path / "demo.json")])
    assert result.exit_code == 0
    assert (tmp_path / "demo.json").exists()
