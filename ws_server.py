# ws_server.py: inference-only backend for frontend integration
# Receives crowd/fire updates, runs model.predict(obs), computes lights,
# and returns per-cell lights_delta.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from stable_baselines3 import PPO

from evac_core import (
    AUTO_ACTION,
    DENSITY_LIMIT,
    FIRE_RADIUS,
    MAX_EXITS,
    N_ACTIONS,
    bfs_distance_map_from_sources,
    build_guidance_corridor_mask,
    light_grid_to_delta,
    manhattan_circle_mask_fast,
)

TTL_MS = 500

# --- New density / anti-flicker logic ---
CONGESTION_RED_ON = max(5, DENSITY_LIMIT + 2)
CONGESTION_RED_OFF = max(4, DENSITY_LIMIT + 1)
CONGESTION_HOLD_TICKS = 3

app = FastAPI()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "evac-light-backend",
        "websocket": "/ws",
    }


MODEL: Optional[PPO] = None
MODEL_META: Optional[Dict[str, Any]] = None


@dataclass
class SessionState:
    session_id: str
    w: int
    h: int
    layout: np.ndarray
    exits: List[Tuple[int, int]]
    dist_nearest: np.ndarray
    dist_per_exit: List[np.ndarray]
    crowd: np.ndarray
    fire: np.ndarray
    light_prev: np.ndarray
    congestion_red_state: np.ndarray
    congestion_hold_until: np.ndarray
    last_t: int


SESSIONS: Dict[str, SessionState] = {}


def load_model_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    meta_path = Path(model_path + ".meta.json")
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_layout(
    walls: List[List[int]],
    exits: List[List[int]],
    w: int,
    h: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    layout = np.zeros((w, h), dtype=np.float32)

    for x, y in walls:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            layout[x, y] = 1.0

    exit_list: List[Tuple[int, int]] = []
    for x, y in exits:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h and layout[x, y] != 1.0:
            layout[x, y] = 2.0
            exit_list.append((x, y))

    if len(exit_list) == 0:
        raise ValueError("INIT must include at least one exit.")

    if len(exit_list) > MAX_EXITS:
        raise ValueError(f"INIT includes {len(exit_list)} exits, but model supports at most {MAX_EXITS} exits.")

    return layout, exit_list


def apply_tick_update(st: SessionState, msg: Dict[str, Any]) -> None:
    w, h = st.w, st.h

    if "crowd_full" in msg:
        g = np.array(msg["crowd_full"], dtype=np.float32)
        if g.shape != (w, h):
            raise ValueError(f"crowd_full must be shape ({w}, {h})")
        st.crowd = g
    else:
        for x, y, c in msg.get("crowd_delta", []):
            x, y = int(x), int(y)
            if 0 <= x < w and 0 <= y < h:
                st.crowd[x, y] = float(c)

    if "fire_full" in msg:
        g = np.array(msg["fire_full"], dtype=np.float32)
        if g.shape != (w, h):
            raise ValueError(f"fire_full must be shape ({w}, {h})")
        st.fire = (g > 0).astype(np.float32)
    else:
        for x, y in msg.get("fire_on", []):
            x, y = int(x), int(y)
            if 0 <= x < w and 0 <= y < h:
                st.fire[x, y] = 1.0
        for x, y in msg.get("fire_off", []):
            x, y = int(x), int(y)
            if 0 <= x < w and 0 <= y < h:
                st.fire[x, y] = 0.0


def update_congestion_state(
    layout: np.ndarray,
    fire: np.ndarray,
    crowd: np.ndarray,
    prev_state: np.ndarray,
    hold_until: np.ndarray,
    tick: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
    crowd: np.ndarray,
    corridor_mask: np.ndarray,
    congestion_red_state: np.ndarray,
) -> np.ndarray:
    w, h = layout.shape
    light = np.zeros((w, h), dtype=np.float32)

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


def choose_action(st: SessionState) -> int:
    if MODEL is None:
        return AUTO_ACTION

    obs = {
        "layout": st.layout.astype(np.float32),
        "fire": st.fire.astype(np.float32),
        "light": st.light_prev.astype(np.float32),
        "crowd": st.crowd.astype(np.float32),
    }

    a, _ = MODEL.predict(obs, deterministic=True)
    a = int(a)

    if a < 0:
        a = 0
    if a >= N_ACTIONS:
        a = AUTO_ACTION

    return a


def compute_light(st: SessionState, action: int, tick: int) -> Tuple[np.ndarray, int, str]:
    if action == AUTO_ACTION or action >= len(st.exits):
        dist_target = st.dist_nearest
        eff = AUTO_ACTION
        mode = "AUTO_NEAREST"
    else:
        dist_target = st.dist_per_exit[action]
        eff = action
        mode = "GUIDE_EXIT"

    corridor = build_guidance_corridor_mask(st.layout, st.fire, st.crowd, dist_target)

    st.congestion_red_state, st.congestion_hold_until = update_congestion_state(
        st.layout,
        st.fire,
        st.crowd,
        st.congestion_red_state,
        st.congestion_hold_until,
        tick,
    )

    light = build_light_field_density_aware(
        st.layout,
        st.fire,
        st.crowd,
        corridor,
        st.congestion_red_state,
    )

    return light, eff, mode


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    current_session_id: Optional[str] = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "init":
                sid = str(msg.get("session_id", "default"))
                current_session_id = sid

                grid_msg = msg.get("grid", {})
                w = int(grid_msg.get("w", 20))
                h = int(grid_msg.get("h", 20))

                if w < 3 or h < 3:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "grid.w and grid.h must both be at least 3",
                    }))
                    continue

                if MODEL_META is not None:
                    model_w = int(MODEL_META.get("width", -1))
                    model_h = int(MODEL_META.get("height", -1))
                    if model_w != w or model_h != h:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "message": f"model/grid mismatch: model expects {model_w}x{model_h}, got {w}x{h}",
                        }))
                        continue

                layout_msg = msg.get("layout", {})
                walls = layout_msg.get("walls", [])
                exits = layout_msg.get("exits", [])

                try:
                    layout, exit_list = parse_layout(walls, exits, w, h)
                    dist_nearest = bfs_distance_map_from_sources(layout, exit_list)
                    dist_per_exit = [bfs_distance_map_from_sources(layout, [ex]) for ex in exit_list]

                    SESSIONS[sid] = SessionState(
                        session_id=sid,
                        w=w,
                        h=h,
                        layout=layout,
                        exits=exit_list,
                        dist_nearest=dist_nearest,
                        dist_per_exit=dist_per_exit,
                        crowd=np.zeros((w, h), dtype=np.float32),
                        fire=np.zeros((w, h), dtype=np.float32),
                        light_prev=np.zeros((w, h), dtype=np.float32),
                        congestion_red_state=np.zeros((w, h), dtype=bool),
                        congestion_hold_until=np.zeros((w, h), dtype=np.int32),
                        last_t=-1,
                    )

                    await ws.send_text(json.dumps({
                        "type": "ack",
                        "message": "init_ok",
                        "session_id": sid,
                    }))
                except Exception as e:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"init_failed: {e}",
                    }))
                continue

            if mtype == "tick":
                sid = str(msg.get("session_id", current_session_id or "default"))
                if sid not in SESSIONS:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "No session. Send init first.",
                    }))
                    continue

                st = SESSIONS[sid]
                t = int(msg.get("t", st.last_t + 1))
                st.last_t = t

                try:
                    apply_tick_update(st, msg)
                    action = choose_action(st)
                    light, eff, mode = compute_light(st, action, t)
                    delta = light_grid_to_delta(st.light_prev, light)
                    st.light_prev = light

                    now_ms = int(time.time() * 1000)
                    out = {
                        "type": "cmd",
                        "t": t,
                        "ts_ms": now_ms,
                        "ttl_ms": TTL_MS,
                        "policy": {"action": int(eff), "mode": mode},
                        "lights_delta": delta,
                        "counts": {
                            "n_white": int(np.sum(light == -1)),
                            "n_red": int(np.sum(light == 1)),
                            "n_congestion_red": int(np.sum(st.congestion_red_state)),
                        },
                    }
                    await ws.send_text(json.dumps(out))
                except Exception as e:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"tick_failed: {e}",
                    }))
                continue

            await ws.send_text(json.dumps({
                "type": "error",
                "message": "Unknown message type.",
            }))

    except WebSocketDisconnect:
        return


def main():
    global MODEL, MODEL_META

    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="evac_light_ppo.zip")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    try:
        MODEL = PPO.load(args.model)
        MODEL_META = load_model_metadata(args.model)
        print(f"Loaded model: {args.model}")
        if MODEL_META is not None:
            print(
                f"Model metadata: width={MODEL_META.get('width')} "
                f"height={MODEL_META.get('height')} "
                f"n_people={MODEL_META.get('n_people')}"
            )
        else:
            print("No model metadata found. Grid/model compatibility checks are disabled.")
    except Exception as e:
        MODEL = None
        MODEL_META = None
        print(f"Model not loaded, using AUTO fallback. Reason: {e}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()