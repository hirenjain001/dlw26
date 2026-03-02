# train_env.py: this script generates random scenarios and learns a policy
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from evac_core import (
    WORLD_W, WORLD_H, DIRS,
    MAX_EXITS, AUTO_ACTION, N_ACTIONS,
    bfs_distance_map_from_sources,
    build_guidance_corridor_mask,
    build_light_field,
)

# --- Training dynamics / reward ---
N_PEOPLE = 120
MAX_STEPS = 200
FIRE_SPREAD_CHANCE = 0.15

LIGHT_ALPHA = 12.0
DENSITY_LIMIT = 7

R_EVAC = 20.0
R_BURN = -50.0
R_STEP = -1.0
R_OVER = -30.0
R_FAIL_TIMEOUT = -500.0
R_NO_GUIDE = -0.25


def make_border_walls() -> np.ndarray:
    layout = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
    layout[0, :] = 1
    layout[WORLD_W - 1, :] = 1
    layout[:, 0] = 1
    layout[:, WORLD_H - 1] = 1
    return layout


def place_random_rect_obstacle(rng: np.random.Generator, layout: np.ndarray):
    w = int(rng.integers(2, 5))
    h = int(rng.integers(2, 5))
    x0 = int(rng.integers(1, WORLD_W - 1 - w))
    y0 = int(rng.integers(1, WORLD_H - 1 - h))
    layout[x0:x0 + w, y0:y0 + h] = 1.0


def generate_training_scenario(seed: int) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    layout = make_border_walls()

    # obstacles
    for _ in range(int(rng.integers(4, 11))):
        place_random_rect_obstacle(rng, layout)

    # exits (1..MAX_EXITS)
    exits: List[Tuple[int, int]] = []
    n_exits = int(rng.integers(1, MAX_EXITS + 1))
    tries = 0
    while len(exits) < n_exits and tries < 500:
        tries += 1
        x = int(rng.integers(1, WORLD_W - 1))
        y = int(rng.integers(1, WORLD_H - 1))
        if layout[x, y] == 0:
            layout[x, y] = 2.0
            exits.append((x, y))

    fire = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
    n_fires = int(rng.integers(1, 4))
    tries = 0
    while np.sum(fire) < n_fires and tries < 500:
        tries += 1
        x = int(rng.integers(1, WORLD_W - 1))
        y = int(rng.integers(1, WORLD_H - 1))
        if layout[x, y] == 0 and fire[x, y] == 0:
            fire[x, y] = 1.0

    return layout, exits, fire


def spawn_people(seed: int, layout: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    crowd = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
    placed = 0
    while placed < N_PEOPLE:
        x = int(rng.integers(0, WORLD_W))
        y = int(rng.integers(0, WORLD_H))
        if layout[x, y] == 0:
            crowd[x, y] += 1.0
            placed += 1
    return crowd


def move_crowd(layout: np.ndarray, fire: np.ndarray, light: np.ndarray, crowd: np.ndarray, dist_map: np.ndarray):
    new = np.zeros_like(crowd, dtype=np.float32)
    evac = 0.0
    for x, y in np.argwhere(crowd > 0):
        count = crowd[x, y]
        best_score = 1e9
        best_pos = (x, y)

        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= WORLD_W or ny < 0 or ny >= WORLD_H:
                continue
            if layout[nx, ny] == 1 or fire[nx, ny] == 1:
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
            evac += float(count)
        else:
            new[bx, by] += count

    return new, evac


def spread_fire(layout: np.ndarray, fire: np.ndarray, seed_for_step: int) -> np.ndarray:
    rng = np.random.default_rng(seed_for_step)
    new_fire = fire.copy()
    for fx, fy in np.argwhere(fire == 1):
        for dx, dy in DIRS:
            nx, ny = fx + dx, fy + dy
            if nx < 0 or nx >= WORLD_W or ny < 0 or ny >= WORLD_H:
                continue
            if layout[nx, ny] != 0:
                continue
            if new_fire[nx, ny] == 1:
                continue
            if rng.random() < FIRE_SPREAD_CHANCE:
                new_fire[nx, ny] = 1.0
    return new_fire


class TrainEnv(gym.Env):
    def __init__(self, seed=0):
        super().__init__()
        self.seed_base = int(seed)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "layout": spaces.Box(low=0.0, high=2.0, shape=(WORLD_W, WORLD_H), dtype=np.float32),
            "fire": spaces.Box(low=0.0, high=1.0, shape=(WORLD_W, WORLD_H), dtype=np.float32),
            "light": spaces.Box(low=-1.0, high=1.0, shape=(WORLD_W, WORLD_H), dtype=np.float32),
            "crowd": spaces.Box(low=0.0, high=float(N_PEOPLE), shape=(WORLD_W, WORLD_H), dtype=np.float32),
        })

        self.layout = make_border_walls()
        self.exits: List[Tuple[int, int]] = []
        self.fire = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
        self.crowd = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
        self.light = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)

        self.dist_nearest = np.full((WORLD_W, WORLD_H), 999, dtype=np.float32)
        self.dist_per_exit: List[np.ndarray] = []

        self.steps = 0

    def _recompute_dist(self):
        self.dist_nearest = bfs_distance_map_from_sources(self.layout, self.exits)
        self.dist_per_exit = [bfs_distance_map_from_sources(self.layout, [ex]) for ex in self.exits]

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        if seed is None:
            seed = int(time.time()) % 1_000_000
        self.seed_base = int(seed)

        self.layout, self.exits, self.fire = generate_training_scenario(self.seed_base)
        self._recompute_dist()
        self.crowd = spawn_people(self.seed_base + 12345, self.layout)
        self.light[:, :] = 0.0

        return {"layout": self.layout, "fire": self.fire, "light": self.light, "crowd": self.crowd}, {}

    def step(self, action: int):
        self.steps += 1
        action = int(action)

        if action == AUTO_ACTION or action >= len(self.exits):
            dist_target = self.dist_nearest
            effective = AUTO_ACTION
            mode = "AUTO_NEAREST"
        else:
            dist_target = self.dist_per_exit[action]
            effective = action
            mode = "GUIDE_EXIT"

        corridor = build_guidance_corridor_mask(self.layout, self.fire, self.crowd, dist_target)
        self.light = build_light_field(self.layout, self.fire, self.crowd, corridor)

        self.crowd, evacuated = move_crowd(self.layout, self.fire, self.light, self.crowd, dist_target)

        step_seed = self.seed_base + self.steps * 9973
        self.fire = spread_fire(self.layout, self.fire, step_seed)

        burned = float(np.sum(self.crowd[self.fire == 1.0]))
        if burned > 0:
            self.crowd[self.fire == 1.0] = 0.0

        excess = np.maximum(0.0, self.crowd - DENSITY_LIMIT)
        overcrowd_cost = float(np.sum(excess))

        reward = (
            R_EVAC * evacuated
            + R_BURN * burned
            + R_STEP
            + R_OVER * overcrowd_cost
            + (R_NO_GUIDE if effective == AUTO_ACTION else 0.0)
        )

        terminated = (np.sum(self.crowd) <= 0.0)
        truncated = (self.steps >= MAX_STEPS)
        if truncated and not terminated:
            reward += R_FAIL_TIMEOUT

        obs = {"layout": self.layout, "fire": self.fire, "light": self.light, "crowd": self.crowd}
        info = {"mode": mode, "effective_action": effective, "remaining": float(np.sum(self.crowd))}
        return obs, float(reward), terminated, truncated, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="evac_light_ppo.zip")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = TrainEnv(seed=args.seed)
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
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model)
    print(f"Saved model to: {args.model}")


if __name__ == "__main__":
    main()