# evac_core.py: contains shared logic
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

# ============================================================
# CORE CONSTANTS
# ============================================================

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

# --- Density / anti-flicker logic ---
DENSITY_COST_SOFT = max(1, DENSITY_LIMIT - 1)
DENSITY_COST_ALPHA = 1.25
DENSITY_COST_POWER = 2.0

CONGESTION_RED_ON = max(5, DENSITY_LIMIT + 2)
CONGESTION_RED_OFF = max(4, DENSITY_LIMIT + 1)
CONGESTION_HOLD_TICKS = 3


# ============================================================
# FAST MASKS
# ============================================================

def manhattan_circle_mask_fast(
    center_cells: np.ndarray,
    shape: Tuple[int, int],
    radius: int,
) -> np.ndarray:
    """
    Build a boolean mask containing all cells within Manhattan distance
    <= radius from any center cell.
    """
    w, h = shape
    if len(center_cells) == 0:
        return np.zeros((w, h), dtype=bool)

    xx, yy = np.ogrid[:w, :h]
    full = np.zeros((w, h), dtype=bool)

    for cx, cy in center_cells:
        dist = np.abs(xx - int(cx)) + np.abs(yy - int(cy))
        full |= (dist <= radius)

    return full


def dilate_mask_manhattan(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Dilate a boolean mask using Manhattan distance.
    """
    if radius <= 0:
        return mask
    centers = np.argwhere(mask)
    return manhattan_circle_mask_fast(centers, mask.shape, radius)


# ============================================================
# BFS DISTANCE MAP (multi-source)
# ============================================================

def bfs_distance_map_from_sources(
    layout: np.ndarray,
    sources: List[Tuple[int, int]],
) -> np.ndarray:
    """
    layout: grid with 0 walkway, 1 wall, 2 exit
    sources: list of (x,y) with distance 0
    returns dist map float32
    """
    w, h = layout.shape
    dist = np.full((w, h), 999, dtype=np.int32)
    q = deque()

    for sx, sy in sources:
        if 0 <= sx < w and 0 <= sy < h and layout[sx, sy] != 1:
            dist[sx, sy] = 0
            q.append((sx, sy))

    if not q:
        return dist.astype(np.float32)

    while q:
        x, y = q.popleft()
        base_d = dist[x, y]

        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if layout[nx, ny] == 1:
                continue
            if dist[nx, ny] != 999:
                continue

            dist[nx, ny] = base_d + 1
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
    w, h = layout.shape
    corridor = np.zeros((w, h), dtype=bool)

    start_cells = np.argwhere(crowd > 0)
    if len(start_cells) == 0:
        return corridor

    # Trace from densest cells only for speed
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
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
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
# LIGHT FIELD
# ============================================================

def build_light_field(
    layout: np.ndarray,
    fire: np.ndarray,
    crowd: np.ndarray,
    corridor_mask: np.ndarray,
) -> np.ndarray:
    """
    Legacy light field:
    returns light grid float32: -1 white, 0 off, +1 red
    """
    light = np.zeros(layout.shape, dtype=np.float32)

    fire_cells = np.argwhere(fire == 1)
    if len(fire_cells) > 0:
        fire_mask = manhattan_circle_mask_fast(fire_cells, layout.shape, FIRE_RADIUS)
        light[np.where(fire_mask & (layout != 1))] = 1.0

    overcrowd_mask = (crowd > DENSITY_LIMIT)
    light[np.where(overcrowd_mask & (layout != 1) & (layout != 2))] = 1.0

    eligible = corridor_mask & (layout != 1) & (layout != 2) & (fire == 0) & (light != 1.0)
    light[np.where(eligible)] = -1.0

    return light


def update_congestion_state(
    layout: np.ndarray,
    fire: np.ndarray,
    crowd: np.ndarray,
    prev_state: np.ndarray,
    hold_until: np.ndarray,
    tick: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hysteresis + hold-time for congestion warning.
    This only affects visual congestion-red cells, not fire-red.
    """
    eligible = (layout != 1) & (layout != 2) & (fire == 0)

    next_state = prev_state.copy()
    next_hold_until = hold_until.copy()

    next_state[~eligible] = False
    next_hold_until[~eligible] = tick

    can_flip = (tick >= next_hold_until)

    turn_on = eligible & (~next_state) & can_flip & (crowd >= CONGESTION_RED_ON)
    turn_off = eligible & next_state & can_flip & (crowd <= CONGESTION_RED_OFF)

    next_state[turn_on] = True
    next_hold_until[turn_on] = tick + CONGESTION_HOLD_TICKS

    next_state[turn_off] = False
    next_hold_until[turn_off] = tick + CONGESTION_HOLD_TICKS

    return next_state, next_hold_until


def build_light_field_density_aware(
    layout: np.ndarray,
    fire: np.ndarray,
    corridor_mask: np.ndarray,
    congestion_red_state: np.ndarray,
) -> np.ndarray:
    """
    Priority:
    1) fire red (hard override)
    2) corridor white
    3) sustained congestion red, but only off-corridor
    """
    light = np.zeros(layout.shape, dtype=np.float32)

    fire_cells = np.argwhere(fire == 1)
    if len(fire_cells) > 0:
        fire_mask = manhattan_circle_mask_fast(fire_cells, layout.shape, FIRE_RADIUS)
        light[np.where(fire_mask & (layout != 1))] = 1.0

    corridor_white = corridor_mask & (layout != 1) & (layout != 2) & (fire == 0) & (light != 1.0)
    light[np.where(corridor_white)] = -1.0

    congestion_red = (
        congestion_red_state
        & (layout != 1)
        & (layout != 2)
        & (fire == 0)
        & (~corridor_mask)
        & (light != 1.0)
    )
    light[np.where(congestion_red)] = 1.0

    return light


def density_penalty(crowd_value: float) -> float:
    excess = max(0.0, float(crowd_value) - float(DENSITY_COST_SOFT))
    return DENSITY_COST_ALPHA * (excess ** DENSITY_COST_POWER)


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
        if np.isclose(v, -1.0):
            out.append([int(x), int(y), "WHITE"])
        elif np.isclose(v, 1.0):
            out.append([int(x), int(y), "RED"])
        else:
            out.append([int(x), int(y), "OFF"])

    return out


# ============================================================
# JSON HELPERS
# ============================================================

def schema_init_example(w: int = 20, h: int = 20) -> Dict[str, Any]:
    return {
        "type": "init",
        "session_id": "demo",
        "grid": {"w": w, "h": h},
        "layout": {
            "walls": [[0, 0], [0, 1]],
            "exits": [[1, 1], [max(1, w - 2), max(1, h - 2)]],
        },
        "opts": {"max_exits": MAX_EXITS},
    }


def schema_tick_example() -> Dict[str, Any]:
    return {
        "type": "tick",
        "session_id": "demo",
        "t": 12,
        "ts_ms": 1700000123456,
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