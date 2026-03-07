# train_env.py: this script generates random scenarios and learns a policy
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from evac_core import (
    AUTO_ACTION,
    DENSITY_LIMIT,
    DIRS,
    MAX_EXITS,
    N_ACTIONS,
    bfs_distance_map_fire_aware,
    build_fire_danger_mask,
    build_guidance_corridor_mask,
    build_light_field_density_aware,
    density_penalty,
    update_congestion_state,
)

# --- Training dynamics / reward ---
DEFAULT_N_PEOPLE = 120
MAX_STEPS = 200
FIRE_SPREAD_CHANCE = 0.15

LIGHT_ALPHA = 12.0

R_EVAC = 20.0
R_BURN = -50.0
R_STEP = -1.0
R_OVER = -30.0
R_FAIL_TIMEOUT = -500.0
R_NO_GUIDE = -0.25


def make_border_walls(w: int, h: int) -> np.ndarray:
    layout = np.zeros((w, h), dtype=np.float32)
    layout[0, :] = 1
    layout[w - 1, :] = 1
    layout[:, 0] = 1
    layout[:, h - 1] = 1
    return layout


def place_random_rect_obstacle(rng: np.random.Generator, layout: np.ndarray) -> None:
    w, h = layout.shape

    max_rect_w = max(3, min(5, w - 2))
    max_rect_h = max(3, min(5, h - 2))

    rect_w = int(rng.integers(2, max_rect_w))
    rect_h = int(rng.integers(2, max_rect_h))

    if w - 1 - rect_w <= 1 or h - 1 - rect_h <= 1:
        return

    x0 = int(rng.integers(1, w - 1 - rect_w))
    y0 = int(rng.integers(1, h - 1 - rect_h))
    layout[x0:x0 + rect_w, y0:y0 + rect_h] = 1.0


def ensure_at_least_one_exit(
    rng: np.random.Generator,
    layout: np.ndarray,
    exits: List[Tuple[int, int]],
) -> None:
    if exits:
        return

    walkable = np.argwhere(layout == 0)
    if len(walkable) == 0:
        raise ValueError("Failed to generate scenario: no walkable cell available for an exit.")

    x, y = walkable[int(rng.integers(0, len(walkable)))]
    layout[x, y] = 2.0
    exits.append((int(x), int(y)))


def generate_training_scenario(
    seed: int,
    w: int,
    h: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    layout = make_border_walls(w, h)

    n_obstacles = int(rng.integers(4, 11))
    for _ in range(n_obstacles):
        place_random_rect_obstacle(rng, layout)

    exits: List[Tuple[int, int]] = []
    n_exits = int(rng.integers(1, MAX_EXITS + 1))
    tries = 0
    while len(exits) < n_exits and tries < 500:
        tries += 1
        x = int(rng.integers(1, w - 1))
        y = int(rng.integers(1, h - 1))
        if layout[x, y] == 0:
            layout[x, y] = 2.0
            exits.append((x, y))

    ensure_at_least_one_exit(rng, layout, exits)

    fire = np.zeros((w, h), dtype=np.float32)
    n_fires = int(rng.integers(1, 4))
    tries = 0
    placed_fires = 0
    while placed_fires < n_fires and tries < 500:
        tries += 1
        x = int(rng.integers(1, w - 1))
        y = int(rng.integers(1, h - 1))
        if layout[x, y] == 0 and fire[x, y] == 0:
            fire[x, y] = 1.0
            placed_fires += 1

    return layout, exits, fire


def spawn_people(seed: int, layout: np.ndarray, n_people: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    crowd = np.zeros(layout.shape, dtype=np.float32)

    walkable = np.argwhere(layout == 0)
    if len(walkable) == 0:
        return crowd

    placed = 0
    while placed < n_people:
        idx = int(rng.integers(0, len(walkable)))
        x, y = walkable[idx]
        crowd[x, y] += 1.0
        placed += 1

    return crowd


def move_crowd(
    layout: np.ndarray,
    fire: np.ndarray,
    light: np.ndarray,
    crowd: np.ndarray,
    dist_map: np.ndarray,
) -> Tuple[np.ndarray, float]:
    w, h = layout.shape
    new = np.zeros_like(crowd, dtype=np.float32)
    evac = 0.0

    danger_mask = build_fire_danger_mask(layout, fire)

    occupied = np.argwhere(crowd > 0)
    if len(occupied) == 0:
        return new, evac

    densities = crowd[occupied[:, 0], occupied[:, 1]]
    order = np.argsort(-densities)
    occupied = occupied[order]

    for x, y in occupied:
        count = float(crowd[x, y])
        if count <= 0:
            continue

        best_score = 1e9
        best_pos = (int(x), int(y))

        for dx, dy in DIRS:
            nx, ny = int(x + dx), int(y + dy)
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if layout[nx, ny] == 1 or danger_mask[nx, ny]:
                continue
            if layout[nx, ny] == 2:
                best_pos = (nx, ny)
                best_score = -1e9
                break

            projected_density = float(crowd[nx, ny] + new[nx, ny] + count)
            score = (
                float(dist_map[nx, ny])
                + (LIGHT_ALPHA * float(light[nx, ny]))
                + density_penalty(projected_density)
            )

            if score < best_score:
                best_score = score
                best_pos = (nx, ny)

        bx, by = best_pos
        if layout[bx, by] == 2:
            evac += count
        else:
            new[bx, by] += count

    return new, evac


def spread_fire(layout: np.ndarray, fire: np.ndarray, seed_for_step: int) -> np.ndarray:
    rng = np.random.default_rng(seed_for_step)
    w, h = layout.shape
    new_fire = fire.copy()

    burning = np.argwhere(fire == 1)
    for fx, fy in burning:
        for dx, dy in DIRS:
            nx, ny = int(fx + dx), int(fy + dy)
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if layout[nx, ny] != 0:
                continue
            if new_fire[nx, ny] == 1:
                continue
            if rng.random() < FIRE_SPREAD_CHANCE:
                new_fire[nx, ny] = 1.0

    return new_fire


class TrainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, width: int = 20, height: int = 20, n_people: int = DEFAULT_N_PEOPLE, seed: int = 0):
        super().__init__()

        self.w = int(width)
        self.h = int(height)
        self.n_people = int(n_people)
        self.seed_base = int(seed)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "layout": spaces.Box(low=0.0, high=2.0, shape=(self.w, self.h), dtype=np.float32),
            "fire": spaces.Box(low=0.0, high=1.0, shape=(self.w, self.h), dtype=np.float32),
            "light": spaces.Box(low=-1.0, high=1.0, shape=(self.w, self.h), dtype=np.float32),
            "crowd": spaces.Box(low=0.0, high=float(self.n_people), shape=(self.w, self.h), dtype=np.float32),
        })

        self.layout = make_border_walls(self.w, self.h)
        self.exits: List[Tuple[int, int]] = []
        self.fire = np.zeros((self.w, self.h), dtype=np.float32)
        self.crowd = np.zeros((self.w, self.h), dtype=np.float32)
        self.light = np.zeros((self.w, self.h), dtype=np.float32)

        self.steps = 0

        self.congestion_red_state = np.zeros((self.w, self.h), dtype=bool)
        self.congestion_hold_until = np.zeros((self.w, self.h), dtype=np.int32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "layout": self.layout.astype(np.float32),
            "fire": self.fire.astype(np.float32),
            "light": self.light.astype(np.float32),
            "crowd": self.crowd.astype(np.float32),
        }

    def _compute_fire_aware_maps(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        dist_nearest = bfs_distance_map_fire_aware(self.layout, self.fire, self.exits)
        dist_per_exit = [bfs_distance_map_fire_aware(self.layout, self.fire, [ex]) for ex in self.exits]
        return dist_nearest, dist_per_exit

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        if seed is None:
            seed = int(time.time() * 1000) % 1_000_000
        self.seed_base = int(seed)

        self.layout, self.exits, self.fire = generate_training_scenario(self.seed_base, self.w, self.h)
        self.crowd = spawn_people(self.seed_base + 12345, self.layout, self.n_people)
        self.light[:, :] = 0.0
        self.congestion_red_state[:, :] = False
        self.congestion_hold_until[:, :] = 0

        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1
        action = int(action)

        dist_nearest, dist_per_exit = self._compute_fire_aware_maps()

        if action == AUTO_ACTION or action >= len(self.exits):
            dist_target = dist_nearest
            effective = AUTO_ACTION
            mode = "AUTO_NEAREST"
        else:
            dist_target = dist_per_exit[action]
            effective = action
            mode = "GUIDE_EXIT"

        corridor = build_guidance_corridor_mask(self.layout, self.fire, self.crowd, dist_target)

        self.congestion_red_state, self.congestion_hold_until = update_congestion_state(
            self.layout,
            self.fire,
            self.crowd,
            self.congestion_red_state,
            self.congestion_hold_until,
            self.steps,
        )

        self.light = build_light_field_density_aware(
            self.layout,
            self.fire,
            corridor,
            self.congestion_red_state,
        )

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

        terminated = bool(np.sum(self.crowd) <= 0.0)
        truncated = bool(self.steps >= MAX_STEPS)

        if truncated and not terminated:
            reward += R_FAIL_TIMEOUT

        obs = self._get_obs()
        info = {
            "mode": mode,
            "effective_action": effective,
            "remaining": float(np.sum(self.crowd)),
            "n_congestion_red": int(np.sum(self.congestion_red_state)),
        }
        return obs, float(reward), terminated, truncated, info


def make_env(rank: int, width: int, height: int, n_people: int, seed: int) -> Callable[[], TrainEnv]:
    def _init() -> TrainEnv:
        return TrainEnv(width=width, height=height, n_people=n_people, seed=seed + rank)
    return _init


def save_model_metadata(model_path: str, width: int, height: int, n_people: int, num_envs: int) -> None:
    meta = {
        "width": int(width),
        "height": int(height),
        "n_people": int(n_people),
        "max_exits": int(MAX_EXITS),
        "n_actions": int(N_ACTIONS),
        "num_envs": int(num_envs),
        "saved_at_unix": int(time.time()),
    }
    with open(model_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="evac_light_ppo.zip")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--width", type=int, default=20)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--n_people", type=int, default=DEFAULT_N_PEOPLE)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--checkpoint_freq", type=int, default=10_000)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    p.add_argument("--tensorboard_log", type=str, default="./tb_logs")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)

    args = p.parse_args()

    if args.width < 3 or args.height < 3:
        raise ValueError("width and height must both be at least 3.")
    if args.num_envs < 1:
        raise ValueError("num_envs must be at least 1.")
    if args.n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if args.run_name is None:
        args.run_name = f"ppo_{args.width}x{args.height}_{int(time.time())}"

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    print(f"Training on device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Grid: {args.width}x{args.height}")
    print(f"People: {args.n_people}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Num envs: {args.num_envs}")
    print(f"Model output: {args.model}")
    print(f"TensorBoard log dir: {args.tensorboard_log}")
    print(f"Run name: {args.run_name}")

    if args.num_envs == 1:
        env = DummyVecEnv([
            make_env(0, args.width, args.height, args.n_people, args.seed)
        ])
    else:
        env = SubprocVecEnv([
            make_env(i, args.width, args.height, args.n_people, args.seed)
            for i in range(args.num_envs)
        ])

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=args.checkpoint_dir,
        name_prefix="evac_light_ppo",
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        device=device,
        tensorboard_log=args.tensorboard_log,
        policy_kwargs=policy_kwargs,
    )

    print("Training started...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        tb_log_name=args.run_name,
    )

    model.save(args.model)
    save_model_metadata(args.model, args.width, args.height, args.n_people, args.num_envs)

    print(f"Saved model to: {args.model}")
    print(f"Saved metadata to: {args.model}.meta.json")


if __name__ == "__main__":
    main()