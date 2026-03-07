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
    FIRE_RADIUS,
    MAX_EXITS,
    N_ACTIONS,
    bfs_distance_map_from_sources,
    build_guidance_corridor_mask,
    manhattan_circle_mask_fast,
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

# --- New density / anti-flicker logic ---
DENSITY_COST_SOFT = max(1, DENSITY_LIMIT - 1)   # starts penalizing routing above this density
DENSITY_COST_ALPHA = 1.25
DENSITY_COST_POWER = 2.0

CONGESTION_RED_ON = max(5, DENSITY_LIMIT + 2)
CONGESTION_RED_OFF = max(4, DENSITY_LIMIT + 1)
CONGESTION_HOLD_TICKS = 3


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


def generate_training_scenario(
    seed: int,
    w: int,
    h: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    layout = make_border_walls(w, h)

    # obstacles
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
    w, h = layout.shape
    crowd = np.zeros((w, h), dtype=np.float32)

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
    This ONLY affects visual congestion-red cells, not fire-red.
    """
    eligible = (layout != 1) & (layout != 2) & (fire == 0)

    next_state = prev_state.copy()
    next_hold_until = hold_until.copy()

    # Cells that are no longer eligible cannot remain visually congestion-red
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
    crowd: np.ndarray,
    corridor_mask: np.ndarray,
    congestion_red_state: np.ndarray,
) -> np.ndarray:
    """
    Priority:
    1) fire red (hard override)
    2) corridor white
    3) sustained congestion red, but ONLY off-corridor to avoid contradictory guidance
    """
    w, h = layout.shape
    light = np.zeros((w, h), dtype=np.float32)

    fire_cells = np.argwhere(fire == 1)
    if len(fire_cells) > 0:
        fire_mask = manhattan_circle_mask_fast(fire_cells, layout.shape, FIRE_RADIUS)
        light[np.where(fire_mask & (layout != 1))] = 1.0

    corridor_white = corridor_mask & (layout != 1) & (layout != 2) & (fire == 0) & (light != 1.0)
    light[np.where(corridor_white)] = -1.0

    # Only show congestion red outside the active white corridor
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

    occupied = np.argwhere(crowd > 0)
    for x, y in occupied:
        count = crowd[x, y]
        best_score = 1e9
        best_pos = (int(x), int(y))

        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if layout[nx, ny] == 1 or fire[nx, ny] == 1:
                continue
            if layout[nx, ny] == 2:
                best_pos = (nx, ny)
                best_score = -1e9
                break

            score = (
                dist_map[nx, ny]
                + (LIGHT_ALPHA * light[nx, ny])
                + density_penalty(crowd[nx, ny])
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
            nx, ny = fx + dx, fy + dy
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

        self.dist_nearest = np.full((self.w, self.h), 999, dtype=np.float32)
        self.dist_per_exit: List[np.ndarray] = []

        self.congestion_red_state = np.zeros((self.w, self.h), dtype=bool)
        self.congestion_hold_until = np.zeros((self.w, self.h), dtype=np.int32)

        self.steps = 0

    def _recompute_dist(self) -> None:
        self.dist_nearest = bfs_distance_map_from_sources(self.layout, self.exits)
        self.dist_per_exit = [bfs_distance_map_from_sources(self.layout, [ex]) for ex in self.exits]

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "layout": self.layout.astype(np.float32),
            "fire": self.fire.astype(np.float32),
            "light": self.light.astype(np.float32),
            "crowd": self.crowd.astype(np.float32),
        }

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        if seed is None:
            seed = int(time.time() * 1000) % 1_000_000
        self.seed_base = int(seed)

        self.layout, self.exits, self.fire = generate_training_scenario(self.seed_base, self.w, self.h)
        self._recompute_dist()
        self.crowd = spawn_people(self.seed_base + 12345, self.layout, self.n_people)
        self.light[:, :] = 0.0
        self.congestion_red_state[:, :] = False
        self.congestion_hold_until[:, :] = 0

        return self._get_obs(), {}

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
            self.crowd,
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

        terminated = (np.sum(self.crowd) <= 0.0)
        truncated = (self.steps >= MAX_STEPS)

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