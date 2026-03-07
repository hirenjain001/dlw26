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
FIRE_SAFETY_RADIUS = 2   # routing avoids fire within this Manhattan radius

DENSITY_LIMIT = 3
GUIDE_TRACE_STEPS = 18   # kept for compatibility, but no longer used as a hard cap
GUIDE_THICKNESS = 1

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

def manhattan_circle_mask_fast(center_cells: np.ndarray, shape: Tuple[int, int], radius: int) -> np.ndarray:
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
    if radius <= 0:
        return mask
    centers = np.argwhere(mask)
    return manhattan_circle_mask_fast(centers, mask.shape, radius)


# ============================================================
# FIRE AWARE ROUTING
# ============================================================

def build_fire_danger_mask(layout: np.ndarray, fire: np.ndarray) -> np.ndarray:
    """
    Builds a mask that blocks routing through fire and its safety halo.
    """
    fire_cells = np.argwhere(fire == 1)
    if len(fire_cells) == 0:
        return np.zeros(layout.shape, dtype=bool)

    return manhattan_circle_mask_fast(fire_cells, layout.shape, FIRE_SAFETY_RADIUS)


def build_fire_aware_layout(layout: np.ndarray, fire: np.ndarray) -> np.ndarray:
    """
    Converts fire + safety halo into temporary walls for pathfinding.
    """
    blocked = build_fire_danger_mask(layout, fire)

    layout_safe = layout.copy()
    layout_safe[blocked] = 1.0

    # exits must remain reachable
    layout_safe[layout == 2] = 2.0

    return layout_safe


# ============================================================
# BFS DISTANCE MAP
# ============================================================

def bfs_distance_map_from_sources(layout: np.ndarray, sources: List[Tuple[int, int]]) -> np.ndarray:
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


def bfs_distance_map_fire_aware(layout: np.ndarray, fire: np.ndarray, sources: List[Tuple[int, int]]) -> np.ndarray:
    """
    BFS that avoids fire and its safety halo.
    """
    safe_layout = build_fire_aware_layout(layout, fire)
    return bfs_distance_map_from_sources(safe_layout, sources)


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
    Build a white-light guidance corridor by tracing from every occupied crowd cell
    downhill on the BFS distance map until the route reaches an exit or gets stuck.

    Note:
    - trace_steps is kept in the signature for compatibility with existing callers,
      but the corridor is no longer artificially capped by a fixed number of steps.
    """

    w, h = layout.shape
    corridor = np.zeros((w, h), dtype=bool)

    danger_mask = build_fire_danger_mask(layout, fire)

    start_cells = np.argwhere(crowd > 0)
    if len(start_cells) == 0:
        return corridor

    for sx, sy in start_cells:
        x, y = int(sx), int(sy)
        visited = set()

        while True:
            if (x, y) in visited:
                break
            visited.add((x, y))

            if layout[x, y] == 1 or danger_mask[x, y]:
                break

            if dist_map_target[x, y] >= 999:
                break

            corridor[x, y] = True

            if layout[x, y] == 2:
                break

            best = None
            best_d = dist_map_target[x, y]

            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy

                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if layout[nx, ny] == 1 or danger_mask[nx, ny]:
                    continue

                nd = dist_map_target[nx, ny]
                if nd < best_d:
                    best_d = nd
                    best = (nx, ny)

            if best is None:
                break

            x, y = best

    if thickness > 0:
        corridor = dilate_mask_manhattan(corridor, thickness)

    return corridor


# ============================================================
# LIGHT FIELD
# ============================================================

def build_light_field_density_aware(
    layout: np.ndarray,
    fire: np.ndarray,
    corridor_mask: np.ndarray,
    congestion_red_state: np.ndarray,
) -> np.ndarray:

    light = np.zeros(layout.shape, dtype=np.float32)

    fire_cells = np.argwhere(fire == 1)
    if len(fire_cells) > 0:
        fire_mask = manhattan_circle_mask_fast(fire_cells, layout.shape, FIRE_RADIUS)
        light[np.where(fire_mask & (layout != 1))] = 1.0

    # Include exits in white lighting so the path visually reaches the exit.
    corridor_white = corridor_mask & (layout != 1) & (fire == 0) & (light != 1.0)
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


def update_congestion_state(
    layout: np.ndarray,
    fire: np.ndarray,
    crowd: np.ndarray,
    prev_state: np.ndarray,
    hold_until: np.ndarray,
    tick: int,
):

    eligible = (layout != 1) & (layout != 2) & (fire == 0)

    next_state = prev_state.copy()
    next_hold_until = hold_until.copy()

    next_state[~eligible] = False
    next_hold_until[~eligible] = tick

    can_flip = tick >= next_hold_until

    turn_on = eligible & (~next_state) & can_flip & (crowd >= CONGESTION_RED_ON)
    turn_off = eligible & next_state & can_flip & (crowd <= CONGESTION_RED_OFF)

    next_state[turn_on] = True
    next_hold_until[turn_on] = tick + CONGESTION_HOLD_TICKS

    next_state[turn_off] = False
    next_hold_until[turn_off] = tick + CONGESTION_HOLD_TICKS

    return next_state, next_hold_until


def density_penalty(crowd_value: float) -> float:
    excess = max(0.0, float(crowd_value) - float(DENSITY_COST_SOFT))
    return DENSITY_COST_ALPHA * (excess ** DENSITY_COST_POWER)


# ============================================================
# DELTA ENCODING
# ============================================================

def light_grid_to_delta(prev: np.ndarray, curr: np.ndarray) -> List[List[Any]]:
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