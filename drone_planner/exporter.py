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
