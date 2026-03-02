# evac_core.py: contains shared logic
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ============================================================
# CORE CONSTANTS (must match frontend grid)
# ============================================================
WORLD_W, WORLD_H = 20, 20
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Lights encoding: -1 WHITE, 0 OFF, +1 RED
FIRE_RADIUS = 1
DENSITY_LIMIT = 3
GUIDE_TRACE_STEPS = 18
GUIDE_THICKNESS = 1

# Action space: choose exit idx 0..MAX_EXITS-1 OR AUTO (nearest)
MAX_EXITS = 3
AUTO_ACTION = MAX_EXITS
N_ACTIONS = MAX_EXITS + 1

XX, YY = np.ogrid[:WORLD_W, :WORLD_H]


# ============================================================
# FAST MASKS
# ============================================================
def manhattan_circle_mask_fast(center_cells: np.ndarray, radius: int) -> np.ndarray:
    if len(center_cells) == 0:
        return np.zeros((WORLD_W, WORLD_H), dtype=bool)
    full = np.zeros((WORLD_W, WORLD_H), dtype=bool)
    for cx, cy in center_cells:
        dist = np.abs(XX - cx) + np.abs(YY - cy)
        full |= (dist <= radius)
    return full


def dilate_mask_manhattan(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    centers = np.argwhere(mask)
    return manhattan_circle_mask_fast(centers, radius)


# ============================================================
# BFS DISTANCE MAP (multi-source)
# ============================================================
def bfs_distance_map_from_sources(layout: np.ndarray, sources: List[Tuple[int, int]]) -> np.ndarray:
    """
    layout: 20x20 (0 walkway, 1 wall, 2 exit)
    sources: list of (x,y) with distance 0
    returns dist map float32
    """
    dist = np.full((WORLD_W, WORLD_H), 999, dtype=np.int32)
    q: List[Tuple[int, int]] = []

    for sx, sy in sources:
        if 0 <= sx < WORLD_W and 0 <= sy < WORLD_H and layout[sx, sy] != 1:
            dist[sx, sy] = 0
            q.append((sx, sy))

    if not q:
        return dist.astype(np.float32)

    while q:
        x, y = q.pop(0)
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= WORLD_W or ny < 0 or ny >= WORLD_H:
                continue
            if layout[nx, ny] == 1:
                continue
            if dist[nx, ny] != 999:
                continue
            dist[nx, ny] = dist[x, y] + 1
            q.append((nx, ny))

    return dist.astype(np.float32)


# ============================================================
# GUIDANCE CORRIDOR
# ============================================================
def build_guidance_corridor_mask(
    layout: np.ndarray,
    fire: np.ndarray,
    crowd: np.ndarray,
    dist_map_target: np.ndarray,
    trace_steps: int = GUIDE_TRACE_STEPS,
    thickness: int = GUIDE_THICKNESS,
) -> np.ndarray:
    """
    Builds a path-like corridor from densest crowd cells downhill on dist_map_target.
    Returns boolean mask.
    """
    corridor = np.zeros((WORLD_W, WORLD_H), dtype=bool)
    start_cells = np.argwhere(crowd > 0)
    if len(start_cells) == 0:
        return corridor

    # trace from densest cells only (speed)
    if len(start_cells) > 60:
        densities = crowd[start_cells[:, 0], start_cells[:, 1]]
        idx = np.argsort(-densities)[:60]
        start_cells = start_cells[idx]

    for sx, sy in start_cells:
        x, y = int(sx), int(sy)
        for _ in range(trace_steps):
            if layout[x, y] == 1 or fire[x, y] == 1:
                break

            corridor[x, y] = True
            if layout[x, y] == 2:
                break

            best = (x, y)
            best_d = dist_map_target[x, y]
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= WORLD_W or ny < 0 or ny >= WORLD_H:
                    continue
                if layout[nx, ny] == 1 or fire[nx, ny] == 1:
                    continue
                if dist_map_target[nx, ny] < best_d:
                    best_d = dist_map_target[nx, ny]
                    best = (nx, ny)

            if best == (x, y):
                break
            x, y = best

    if thickness > 0:
        corridor = dilate_mask_manhattan(corridor, thickness)

    return corridor


# ============================================================
# LIGHT FIELD (forced red + corridor white)
# ============================================================
def build_light_field(layout: np.ndarray, fire: np.ndarray, crowd: np.ndarray, corridor_mask: np.ndarray) -> np.ndarray:
    """
    returns light grid float32: -1 white, 0 off, +1 red
    """
    light = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)

    # forced red near fire
    fire_cells = np.argwhere(fire == 1)
    if len(fire_cells) > 0:
        fire_mask = manhattan_circle_mask_fast(fire_cells, FIRE_RADIUS)
        light[np.where(fire_mask & (layout != 1))] = 1.0

    # forced red for overcrowding
    overcrowd_mask = (crowd > DENSITY_LIMIT)
    light[np.where(overcrowd_mask & (layout != 1) & (layout != 2))] = 1.0

    # corridor white (never overwrite forced red)
    eligible = corridor_mask & (layout != 1) & (layout != 2) & (fire == 0) & (light != 1.0)
    light[np.where(eligible)] = -1.0
    return light


# ============================================================
# DELTA ENCODING FOR PER-CELL FRONTEND
# ============================================================
def light_grid_to_delta(prev: np.ndarray, curr: np.ndarray) -> List[List[Any]]:
    """
    prev/curr: -1/0/+1
    -> [[x,y,"WHITE"/"RED"/"OFF"], ...] only changed cells
    """
    changes = np.argwhere(prev != curr)
    out: List[List[Any]] = []
    for x, y in changes:
        v = curr[x, y]
        if v == -1:
            out.append([int(x), int(y), "WHITE"])
        elif v == 1:
            out.append([int(x), int(y), "RED"])
        else:
            out.append([int(x), int(y), "OFF"])
    return out


# ============================================================
# JSON HELPERS (schema you can hand to frontend)
# ============================================================
def schema_init_example() -> Dict[str, Any]:
    return {
        "type": "init",
        "session_id": "demo",
        "grid": {"w": WORLD_W, "h": WORLD_H},
        "layout": {
            "walls": [[0, 0], [0, 1]],
            "exits": [[1, 1], [18, 18]],
        },
        "opts": {"max_exits": MAX_EXITS},
    }


def schema_tick_example() -> Dict[str, Any]:
    return {
        "type": "tick",
        "session_id": "demo",
        "t": 12,
        "ts_ms": 1700000123456,
        # Recommended delta format:
        "crowd_delta": [[3, 7, 2], [3, 8, 5]],
        "fire_on": [[10, 10]],
        "fire_off": [],
        "ack_last_cmd": 11,
    }


def schema_cmd_example() -> Dict[str, Any]:
    return {
        "type": "cmd",
        "t": 12,
        "ts_ms": 1700000123500,
        "ttl_ms": 500,
        "policy": {"action": 0, "mode": "GUIDE_EXIT"},
        "lights_delta": [[3, 7, "WHITE"], [10, 10, "RED"], [4, 4, "OFF"]],
        "counts": {"n_white": 42, "n_red": 16},
    }