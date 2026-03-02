# This script contains past versions of code that is useful but not used for final version
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# CONFIG
# ============================================================
WORLD_W, WORLD_H = 20, 20
N_PEOPLE = 120
MAX_STEPS = 200

LIGHT_ALPHA = 12.0
FIRE_RADIUS = 1
DENSITY_LIMIT = 3
FIRE_SPREAD_CHANCE = 0.15

R_EVAC = 20.0
R_BURN = -50.0
R_STEP = -1.0
R_OVER = -30.0
R_FAIL_TIMEOUT = -500.0

# discourage "do nothing"
R_NO_GUIDE = -0.25

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
XX, YY = np.ogrid[:WORLD_W, :WORLD_H]

# ============================================================
# SCENARIO RANDOMIZATION (versatility)
# ============================================================
MAX_EXITS = 3            # IMPORTANT: fixed max => fixed action space for PPO
MIN_EXITS = 1

MAX_FIRES = 3
MIN_FIRES = 1

MIN_OBSTACLES, MAX_OBSTACLES = 4, 10
OBS_W_RANGE = (2, 5)     # width 2..4
OBS_H_RANGE = (2, 5)     # height 2..4
PLACEMENT_TRIES = 500

# Path-corridor visualization / control
GUIDE_TRACE_STEPS = 18   # how far to trace a corridor from crowd toward exit
GUIDE_THICKNESS = 1      # 0=single-cell line, 1=thicker corridor (manhattan radius)


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
    """Fast-ish dilation using manhattan circles around True cells."""
    if radius <= 0:
        return mask
    centers = np.argwhere(mask)
    return manhattan_circle_mask_fast(centers, radius)


# ============================================================
# LAYOUT + SCENARIO
# ============================================================
def make_empty_layout_with_border_walls() -> np.ndarray:
    layout = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
    layout[0, :] = 1
    layout[WORLD_W - 1, :] = 1
    layout[:, 0] = 1
    layout[:, WORLD_H - 1] = 1
    return layout


def sample_empty_cell(rng: np.random.Generator, layout: np.ndarray) -> Tuple[int, int]:
    for _ in range(PLACEMENT_TRIES):
        x = int(rng.integers(1, WORLD_W - 1))
        y = int(rng.integers(1, WORLD_H - 1))
        if layout[x, y] == 0:
            return x, y
    return 1, 1


def place_obstacle_rect(rng: np.random.Generator, layout: np.ndarray) -> None:
    w = int(rng.integers(*OBS_W_RANGE))
    h = int(rng.integers(*OBS_H_RANGE))
    x0 = int(rng.integers(1, WORLD_W - 1 - w))
    y0 = int(rng.integers(1, WORLD_H - 1 - h))
    layout[x0:x0 + w, y0:y0 + h] = 1


def generate_random_scenario(seed: int) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    layout = make_empty_layout_with_border_walls()

    # obstacles
    for _ in range(int(rng.integers(MIN_OBSTACLES, MAX_OBSTACLES + 1))):
        place_obstacle_rect(rng, layout)

    # exits (variable count, capped)
    exits: List[Tuple[int, int]] = []
    n_exits = int(rng.integers(MIN_EXITS, MAX_EXITS + 1))
    for _ in range(n_exits):
        x, y = sample_empty_cell(rng, layout)
        layout[x, y] = 2
        exits.append((x, y))

    # fire starts (variable count)
    fire = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
    n_fires = int(rng.integers(MIN_FIRES, MAX_FIRES + 1))
    for _ in range(n_fires):
        for _try in range(PLACEMENT_TRIES):
            x, y = sample_empty_cell(rng, layout)
            if layout[x, y] == 0 and fire[x, y] == 0:
                fire[x, y] = 1.0
                break

    return layout, exits, fire


# ============================================================
# DISTANCE MAPS (single-exit + multi-exit)
# ============================================================
def bfs_distance_map_from_sources(layout: np.ndarray, sources: List[Tuple[int, int]]) -> np.ndarray:
    dist = np.full((WORLD_W, WORLD_H), 999, dtype=np.int32)
    q: List[Tuple[int, int]] = []

    for sx, sy in sources:
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


def dist_map_nearest_exit(layout: np.ndarray, exits: List[Tuple[int, int]]) -> np.ndarray:
    return bfs_distance_map_from_sources(layout, exits)


def dist_map_for_exit(layout: np.ndarray, exit_xy: Tuple[int, int]) -> np.ndarray:
    return bfs_distance_map_from_sources(layout, [exit_xy])


# ============================================================
# CROWD + FIRE
# ============================================================
def move_crowd(layout: np.ndarray,
              fire: np.ndarray,
              light: np.ndarray,
              crowd: np.ndarray,
              dist_map: np.ndarray) -> Tuple[np.ndarray, float]:
    new_crowd = np.zeros_like(crowd, dtype=np.float32)
    evacuated = 0.0

    occupied = np.argwhere(crowd > 0)
    for x, y in occupied:
        count = crowd[x, y]
        best_score = 1e9
        best_pos = (x, y)

        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= WORLD_W or ny < 0 or ny >= WORLD_H:
                continue
            if layout[nx, ny] == 1:
                continue
            if fire[nx, ny] == 1:
                continue

            if layout[nx, ny] == 2:
                best_pos = (nx, ny)
                best_score = -1e9
                break

            score = dist_map[nx, ny] + (LIGHT_ALPHA * light[nx, ny])
            if score < best_score:
                best_score = score
                best_pos = (nx, ny)

        bx, by = best_pos
        if layout[bx, by] == 2:
            evacuated += float(count)
        else:
            new_crowd[bx, by] += count

    return new_crowd, evacuated


def spread_fire_deterministic(layout: np.ndarray,
                             fire: np.ndarray,
                             spread_chance: float,
                             seed_for_step: int) -> np.ndarray:
    rng = np.random.default_rng(seed_for_step)
    new_fire = fire.copy()
    fire_cells = np.argwhere(fire == 1)

    for fx, fy in fire_cells:
        for dx, dy in DIRS:
            nx, ny = fx + dx, fy + dy
            if nx < 0 or nx >= WORLD_W or ny < 0 or ny >= WORLD_H:
                continue
            # spread only to empty walkways (exits are protected)
            if layout[nx, ny] != 0:
                continue
            if new_fire[nx, ny] == 1:
                continue
            if rng.random() < spread_chance:
                new_fire[nx, ny] = 1.0

    return new_fire


# ============================================================
# GUIDANCE: "path corridor" instead of "quadrant"
# ============================================================
def build_guidance_corridor_mask(layout: np.ndarray,
                                 fire: np.ndarray,
                                 crowd: np.ndarray,
                                 dist_map_target: np.ndarray,
                                 trace_steps: int = GUIDE_TRACE_STEPS,
                                 thickness: int = GUIDE_THICKNESS) -> np.ndarray:
    """
    Produce a boolean mask of corridor cells by tracing from crowd cells downhill
    on dist_map_target (shortest-path style). This creates path-like white lights.
    """
    corridor = np.zeros((WORLD_W, WORLD_H), dtype=bool)

    # Start tracing from occupied cells (more people => more influence)
    start_cells = np.argwhere(crowd > 0)
    if len(start_cells) == 0:
        return corridor

    # Optionally focus on top-k densest starts to reduce cost
    if len(start_cells) > 60:
        # pick densest cells
        densities = crowd[start_cells[:, 0], start_cells[:, 1]]
        idx = np.argsort(-densities)[:60]
        start_cells = start_cells[idx]

    for sx, sy in start_cells:
        x, y = int(sx), int(sy)
        for _ in range(trace_steps):
            if layout[x, y] == 1:
                break
            if fire[x, y] == 1:
                break
            if layout[x, y] == 2:
                corridor[x, y] = True
                break

            corridor[x, y] = True

            # move to best neighbor by dist (downhill)
            best = (x, y)
            best_d = dist_map_target[x, y]
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= WORLD_W or ny < 0 or ny >= WORLD_H:
                    continue
                if layout[nx, ny] == 1:
                    continue
                if fire[nx, ny] == 1:
                    continue
                if dist_map_target[nx, ny] < best_d:
                    best_d = dist_map_target[nx, ny]
                    best = (nx, ny)

            if best == (x, y):
                break  # stuck (no downhill path)
            x, y = best

    if thickness > 0:
        corridor = dilate_mask_manhattan(corridor, thickness)

    return corridor


def build_light_field(layout: np.ndarray,
                      fire: np.ndarray,
                      crowd: np.ndarray,
                      corridor_mask: np.ndarray) -> np.ndarray:
    """
    Final light values:
      +1 RED: forced near fire and overcrowding
      -1 WHITE: corridor guidance (does NOT override forced red)
    """
    light = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)

    # Forced RED near fire
    fire_cells = np.argwhere(fire == 1)
    if len(fire_cells) > 0:
        fire_mask = manhattan_circle_mask_fast(fire_cells, FIRE_RADIUS)
        light[np.where(fire_mask & (layout != 1))] = 1.0

    # Forced RED for overcrowding
    overcrowd_mask = (crowd > DENSITY_LIMIT)
    light[np.where(overcrowd_mask & (layout != 1) & (layout != 2))] = 1.0

    # Corridor WHITE (only where safe and not forced red)
    eligible = corridor_mask & (layout != 1) & (layout != 2) & (fire == 0) & (light != 1.0)
    light[np.where(eligible)] = -1.0

    return light


# ============================================================
# ACTION SPACE: choose which exit to guide toward (+ "AUTO")
# ============================================================
# actions:
#   0..MAX_EXITS-1 => guide toward that exit index (if exists)
#   MAX_EXITS      => AUTO (nearest-exit gradient, multi-source)
AUTO_ACTION = MAX_EXITS
N_ACTIONS = MAX_EXITS + 1


# ============================================================
# ENVIRONMENT
# ============================================================
class EvacLightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0):
        super().__init__()
        self.seed_base = int(seed)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "layout": spaces.Box(low=0.0, high=2.0, shape=(WORLD_W, WORLD_H), dtype=np.float32),
            "fire":   spaces.Box(low=0.0, high=1.0, shape=(WORLD_W, WORLD_H), dtype=np.float32),
            "light":  spaces.Box(low=-1.0, high=1.0, shape=(WORLD_W, WORLD_H), dtype=np.float32),
            "crowd":  spaces.Box(low=0.0, high=float(N_PEOPLE), shape=(WORLD_W, WORLD_H), dtype=np.float32),
        })

        self.layout = make_empty_layout_with_border_walls()
        self.exits: List[Tuple[int, int]] = []
        self.fire = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
        self.crowd = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
        self.light = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)

        self.dist_nearest = np.full((WORLD_W, WORLD_H), 999, dtype=np.float32)
        self.dist_per_exit: List[np.ndarray] = []

        self.steps = 0

    def _spawn_people(self, seed: int):
        rng = np.random.default_rng(seed)
        self.crowd[:, :] = 0.0
        placed = 0
        while placed < N_PEOPLE:
            x = int(rng.integers(0, WORLD_W))
            y = int(rng.integers(0, WORLD_H))
            if self.layout[x, y] == 0:
                self.crowd[x, y] += 1.0
                placed += 1

    def _recompute_dist_maps(self):
        self.dist_nearest = dist_map_nearest_exit(self.layout, self.exits)
        self.dist_per_exit = [dist_map_for_exit(self.layout, ex) for ex in self.exits]

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {"layout": self.layout, "fire": self.fire, "light": self.light, "crowd": self.crowd}

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        if seed is None:
            seed = int(time.time()) % 1_000_000
        self.seed_base = int(seed)

        # New scenario each episode
        self.layout, self.exits, self.fire = generate_random_scenario(seed=self.seed_base)
        self._recompute_dist_maps()
        self._spawn_people(seed=self.seed_base + 12345)

        # default: no corridor
        self.light[:, :] = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1
        action = int(action)

        # Select dist map target based on action
        if action == AUTO_ACTION or action >= len(self.exits):
            dist_target = self.dist_nearest
            effective_action = AUTO_ACTION
        else:
            dist_target = self.dist_per_exit[action]
            effective_action = action

        # Build corridor -> light field
        corridor = build_guidance_corridor_mask(self.layout, self.fire, self.crowd, dist_target)
        self.light = build_light_field(self.layout, self.fire, self.crowd, corridor)

        # Move crowd
        self.crowd, evacuated = move_crowd(self.layout, self.fire, self.light, self.crowd, dist_target)

        # Fire spread
        step_seed = self.seed_base + self.steps * 9973
        self.fire = spread_fire_deterministic(self.layout, self.fire, FIRE_SPREAD_CHANCE, seed_for_step=step_seed)

        # Burned
        burned = float(np.sum(self.crowd[self.fire == 1.0]))
        if burned > 0:
            self.crowd[self.fire == 1.0] = 0.0

        # Overcrowding
        excess = np.maximum(0.0, self.crowd - DENSITY_LIMIT)
        overcrowd_cost = float(np.sum(excess))

        # Reward
        reward = 0.0
        reward += R_EVAC * float(evacuated)
        reward += R_BURN * burned
        reward += R_STEP
        reward += R_OVER * overcrowd_cost
        if effective_action == AUTO_ACTION:
            reward += R_NO_GUIDE

        terminated = (np.sum(self.crowd) <= 0.0)
        truncated = (self.steps >= MAX_STEPS)
        if truncated and not terminated:
            reward += R_FAIL_TIMEOUT

        info = {
            "evacuated": float(evacuated),
            "burned": burned,
            "overcrowd_excess": overcrowd_cost,
            "remaining": float(np.sum(self.crowd)),
            "steps": self.steps,
            "seed_base": self.seed_base,
            "action": action,
            "effective_action": effective_action,
            "n_exits": int(np.sum(self.layout == 2)),
            "n_fire_cells": int(np.sum(self.fire == 1)),
            "corridor_cells": int(np.sum(corridor)),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


# ============================================================
# TRAINING + INFERENCE
# ============================================================
def train_and_save(model_path="evac_light_ppo.zip", total_timesteps=200_000, seed=0) -> PPO:
    env = EvacLightEnv(seed=seed)
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
    )
    print("Training started...")
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"Saved model to: {model_path}")
    return model


def infer_action(model: PPO, obs: Dict[str, np.ndarray]) -> int:
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


# ============================================================
# FRONTEND CONTRACT HELPERS
# ============================================================
def action_to_payload_exit_choice(action: int) -> Dict[str, str]:
    """
    Simple payload to match your team’s “dummy endpoint” style.
    Here action selects exit index or AUTO.
    """
    if action == AUTO_ACTION:
        return {"mode": "AUTO_NEAREST_EXIT"}
    return {"mode": "GUIDE_TO_EXIT_INDEX", "exit_index": str(action)}


def mask_to_zone_payload(mask_white: np.ndarray) -> Dict[str, str]:
    """
    Optional: if frontend only supports "zones", compress corridor mask into 4 zones.
    Picks the zone with most white cells as WHITE.
    """
    zones = {
        0: (0, 10, 0, 10),
        1: (0, 10, 10, 20),
        2: (10, 20, 0, 10),
        3: (10, 20, 10, 20),
    }
    counts = {}
    for k, (x0, x1, y0, y1) in zones.items():
        counts[k] = int(np.sum(mask_white[x0:x1, y0:y1]))
    best = max(counts, key=lambda z: counts[z])
    payload = {f"zone_{i}": "OFF" for i in range(4)}
    if counts[best] > 0:
        payload[f"zone_{best}"] = "WHITE"
    return payload


# ============================================================
# VISUALIZATION
# ============================================================
def state_to_rgb(layout: np.ndarray, fire: np.ndarray, light: np.ndarray, crowd: np.ndarray) -> np.ndarray:
    img = np.zeros((WORLD_W, WORLD_H, 3), dtype=np.float32)

    img[layout == 1] = [0.05, 0.05, 0.05]  # walls
    img[layout == 2] = [0.0, 1.0, 0.0]     # exits

    red_mask = (light == 1)
    img[red_mask] = [1.0, 0.0, 0.0]

    fire_mask = (fire == 1)
    img[fire_mask] = [1.0, 0.55, 0.0]

    # crowd overlay (blue)
    m = float(np.max(crowd))
    if m > 0:
        crowd_norm = crowd / m
        img[..., 2] = np.maximum(img[..., 2], crowd_norm)

    # semi-transparent white corridor
    white_mask = (light == -1)
    if np.any(white_mask):
        img[white_mask] = img[white_mask] * 0.6 + np.array([1.0, 1.0, 1.0]) * 0.4

    return np.clip(img, 0.0, 1.0)


@dataclass
class EpisodeResult:
    frames: List[np.ndarray]
    actions: List[int]
    infos: List[Dict]


def rollout_episode(env: EvacLightEnv, policy_fn: Callable[[Dict[str, np.ndarray]], int], seed: int) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    frames: List[np.ndarray] = []
    actions: List[int] = []
    infos: List[Dict] = []

    for _ in range(MAX_STEPS):
        a = policy_fn(obs)
        obs, _, term, trunc, info = env.step(a)
        frames.append(state_to_rgb(env.layout, env.fire, env.light, env.crowd))
        actions.append(int(info["effective_action"]))
        infos.append(info)
        if term or trunc:
            break

    return EpisodeResult(frames=frames, actions=actions, infos=infos)


def save_gif(frames: List[np.ndarray], filename: str, fps: int = 10, title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0], interpolation="nearest")

    def update(i):
        im.set_array(frames[i])
        ax.set_title(title if title else f"Step {i}", fontsize=10)
        return [im]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[Saved GIF] {filename} ({len(frames)} frames)")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    MODEL_PATH = "evac_light_ppo.zip"

    model = train_and_save(model_path=MODEL_PATH, total_timesteps=200_000, seed=0)

    env = EvacLightEnv(seed=0)

    def pi_rl(o): return infer_action(model, o)

    ep = rollout_episode(env, pi_rl, seed=123)

    unique, counts = np.unique(np.array(ep.actions), return_counts=True)
    print("[Effective action histogram]", dict(zip(unique.tolist(), counts.tolist())))
    print("[Last info]", ep.infos[-1])

    save_gif(ep.frames, "replay_RL_corridor_variable_seed123.gif", fps=10,
             title="RL corridor guidance | variable exits/fires/obstacles")

    # Example: what you'd send to frontend
    obs, _ = env.reset(seed=123)
    action = infer_action(model, obs)
    print("[Sample payload exit-choice]", json.dumps(action_to_payload_exit_choice(action)))