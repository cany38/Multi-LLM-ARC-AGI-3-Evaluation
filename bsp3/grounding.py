from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from bsp3.config import load_experiment_config

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = _NEIGHBORS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _background_color(frame: np.ndarray) -> int:
    values, counts = np.unique(frame, return_counts=True)
    return int(values[int(np.argmax(counts))])


def extract_objects(
    frame: np.ndarray,
    *,
    connectivity: int = 4,
    background: int | None = None,
    max_objects: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    grid = np.asarray(frame)
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D frame, got shape {grid.shape}")

    h, w = grid.shape
    bg = _background_color(grid) if background is None else int(background)
    deltas = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    visited = np.zeros((h, w), dtype=bool)
    raw_objects: list[dict[str, Any]] = []

    ys, xs = np.where(grid != bg)
    for start_y_np, start_x_np in zip(ys, xs):
        start_y, start_x = int(start_y_np), int(start_x_np)
        if visited[start_y, start_x]:
            continue
        color = int(grid[start_y, start_x])
        visited[start_y, start_x] = True
        queue = deque([(start_y, start_x)])
        min_x = max_x = start_x
        min_y = max_y = start_y
        sum_x = sum_y = area = 0

        while queue:
            y, x = queue.popleft()
            area += 1
            sum_x += x
            sum_y += y
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

            for dy, dx in deltas:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and not visited[ny, nx]
                    and int(grid[ny, nx]) == color
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        raw_objects.append(
            {
                "color": color,
                "area": area,
                "bbox": [min_x, min_y, max_x, max_y],
                "center": [round(sum_x / area, 1), round(sum_y / area, 1)],
            }
        )

    raw_objects.sort(key=lambda o: (o["color"], o["bbox"][1], o["bbox"][0], o["area"]))
    if max_objects is not None and len(raw_objects) > max_objects:
        raw_objects = raw_objects[:max_objects]

    counts_by_color: dict[int, int] = {}
    objects: list[dict[str, Any]] = []
    for obj in raw_objects:
        c = obj["color"]
        rank = counts_by_color.get(c, 0) + 1
        counts_by_color[c] = rank
        objects.append({"id": f"c{c}_{rank}", **obj})
    return objects, bg


def _match_same_slot(
    prev_objs: list[dict[str, Any]],
    curr_objs: list[dict[str, Any]],
    prev_used: set[int],
    curr_used: set[int],
    *,
    require_same_color: bool,
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for ci, curr in enumerate(curr_objs):
        if ci in curr_used:
            continue
        for pi, prev in enumerate(prev_objs):
            if pi in prev_used:
                continue
            if prev["bbox"] != curr["bbox"] or prev["area"] != curr["area"]:
                continue
            if require_same_color and prev["color"] != curr["color"]:
                continue
            if not require_same_color and prev["color"] == curr["color"]:
                continue
            prev_used.add(pi)
            curr_used.add(ci)
            matches.append((pi, ci))
            break
    return matches


def _diff_objects(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_objs: list[dict[str, Any]],
    curr_objs: list[dict[str, Any]],
) -> dict[str, Any]:
    prev_used: set[int] = set()
    curr_used: set[int] = set()

    _match_same_slot(prev_objs, curr_objs, prev_used, curr_used, require_same_color=True)
    recolored_pairs = _match_same_slot(
        prev_objs, curr_objs, prev_used, curr_used, require_same_color=False
    )
    recolored = [
        {
            "id": curr_objs[ci]["id"],
            "bbox": curr_objs[ci]["bbox"],
            "area": curr_objs[ci]["area"],
            "from_color": prev_objs[pi]["color"],
            "to_color": curr_objs[ci]["color"],
        }
        for pi, ci in recolored_pairs
    ]

    # Match remaining objects with equal (color, area) by nearest center to detect translation.
    candidates: list[tuple[float, int, int]] = []
    for ci, curr in enumerate(curr_objs):
        if ci in curr_used:
            continue
        for pi, prev in enumerate(prev_objs):
            if pi in prev_used:
                continue
            if prev["color"] == curr["color"] and prev["area"] == curr["area"]:
                dx = curr["center"][0] - prev["center"][0]
                dy = curr["center"][1] - prev["center"][1]
                candidates.append((dx * dx + dy * dy, pi, ci))

    candidates.sort()
    moved: list[dict[str, Any]] = []
    for _, pi, ci in candidates:
        if pi in prev_used or ci in curr_used:
            continue
        prev_used.add(pi)
        curr_used.add(ci)
        prev = prev_objs[pi]
        curr = curr_objs[ci]
        moved.append(
            {
                "id": curr["id"],
                "color": curr["color"],
                "area": curr["area"],
                "from": prev["center"],
                "to": curr["center"],
                "dx": round(curr["center"][0] - prev["center"][0], 1),
                "dy": round(curr["center"][1] - prev["center"][1], 1),
            }
        )

    appeared = [o for ci, o in enumerate(curr_objs) if ci not in curr_used]
    disappeared = [o for pi, o in enumerate(prev_objs) if pi not in prev_used]
    changed_pixels = int(np.count_nonzero(np.asarray(curr_frame) != np.asarray(prev_frame)))

    return {
        "changed_pixels": changed_pixels,
        "moved": moved,
        "recolored": recolored,
        "appeared": appeared,
        "disappeared": disappeared,
    }


def describe(
    frame: np.ndarray,
    previous_frame: np.ndarray | None = None,
    *,
    connectivity: int = 4,
    max_objects: int | None = 64,
) -> dict[str, Any]:
    """Turn a grid frame (and optional previous frame) into JSON-ready symbols."""
    curr = np.asarray(frame)
    curr_objs, bg = extract_objects(
        curr, connectivity=connectivity, max_objects=max_objects
    )
    colors = sorted(int(c) for c in np.unique(curr))
    diff: dict[str, Any] | None = None
    if previous_frame is not None:
        prev = np.asarray(previous_frame)
        prev_objs, _ = extract_objects(
            prev, connectivity=connectivity, max_objects=max_objects
        )
        diff = _diff_objects(prev, curr, prev_objs, curr_objs)

    h, w = curr.shape
    return {
        "shape": [int(h), int(w)],
        "background_color": bg,
        "colors": colors,
        "objects": curr_objs,
        "diff": diff,
    }


class Grounding:
    """Configured frame-to-JSON symbolic grounder."""

    def __init__(self, *, connectivity: int = 4, max_objects: int = 64) -> None:
        self.connectivity = connectivity
        self.max_objects = max_objects

    @classmethod
    def from_config(cls, config_path: Path | str = "configs/experiment.toml") -> "Grounding":
        cfg = load_experiment_config(config_path).get("grounding", {})
        return cls(
            connectivity=int(cfg.get("connectivity", 4)),
            max_objects=int(cfg.get("max_objects", 64)),
        )

    def describe(
        self, frame: np.ndarray, previous_frame: np.ndarray | None = None
    ) -> dict[str, Any]:
        return describe(
            frame,
            previous_frame,
            connectivity=self.connectivity,
            max_objects=self.max_objects,
        )
