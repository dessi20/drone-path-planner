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
