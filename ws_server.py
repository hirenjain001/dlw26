# ws_server.py: inference-only backend for frontend integration
# Note: This does not simulatie movement or fine, it only receives crowd/fire updates, runs model.predict(obs), computes lights (corridor+forced red), returns per-cell lights_delta
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from stable_baselines3 import PPO

from evac_core import (
    WORLD_W, WORLD_H,
    MAX_EXITS, AUTO_ACTION, N_ACTIONS,
    bfs_distance_map_from_sources,
    build_guidance_corridor_mask,
    build_light_field,
    light_grid_to_delta,
)

# ============================================================
# JSON SCHEMA (what you give to frontend)
# ============================================================
# INIT (once)
# {
#   "type":"init",
#   "session_id":"demo",
#   "grid":{"w":20,"h":20},
#   "layout":{"walls":[[x,y],...], "exits":[[x,y],...]},
#   "opts":{"max_exits":3}
# }
#
# TICK (repeated)
# {
#   "type":"tick",
#   "session_id":"demo",
#   "t":184,
#   "ts_ms":...,
#   "crowd_delta":[[x,y,count],...],
#   "fire_on":[[x,y],...],
#   "fire_off":[[x,y],...]
# }
#
# CMD (reply)
# {
#   "type":"cmd",
#   "t":184,
#   "ts_ms":...,
#   "ttl_ms":500,
#   "policy":{"action":2,"mode":"GUIDE_EXIT"},
#   "lights_delta":[[x,y,"WHITE"],[x,y,"RED"],[x,y,"OFF"],...],
#   "counts":{"n_white":42,"n_red":16}
# }

TTL_MS = 500

app = FastAPI()

# Health / sanity check route
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "evac-light-backend",
        "websocket": "/ws"
    }

MODEL: Optional[PPO] = None


@dataclass
class SessionState:
    session_id: str
    layout: np.ndarray                 # 20x20 (0,1,2)
    exits: List[Tuple[int, int]]
    dist_nearest: np.ndarray           # 20x20
    dist_per_exit: List[np.ndarray]    # list of 20x20
    crowd: np.ndarray                  # 20x20
    fire: np.ndarray                   # 20x20
    light_prev: np.ndarray             # 20x20
    last_t: int


SESSIONS: Dict[str, SessionState] = {}


def parse_layout(walls: List[List[int]], exits: List[List[int]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    layout = np.zeros((WORLD_W, WORLD_H), dtype=np.float32)
    for x, y in walls:
        if 0 <= x < WORLD_W and 0 <= y < WORLD_H:
            layout[int(x), int(y)] = 1.0

    exit_list: List[Tuple[int, int]] = []
    for x, y in exits:
        x, y = int(x), int(y)
        if 0 <= x < WORLD_W and 0 <= y < WORLD_H and layout[x, y] != 1.0:
            layout[x, y] = 2.0
            exit_list.append((x, y))

    if len(exit_list) == 0:
        raise ValueError("INIT must include at least one exit.")
    return layout, exit_list


def apply_tick_update(st: SessionState, msg: Dict[str, Any]) -> None:
    # crowd: delta or full
    if "crowd_full" in msg:
        g = np.array(msg["crowd_full"], dtype=np.float32)
        if g.shape != (WORLD_W, WORLD_H):
            raise ValueError("crowd_full must be 20x20")
        st.crowd = g
    else:
        for x, y, c in msg.get("crowd_delta", []):
            x, y = int(x), int(y)
            if 0 <= x < WORLD_W and 0 <= y < WORLD_H:
                st.crowd[x, y] = float(c)

    # fire: delta or full
    if "fire_full" in msg:
        g = np.array(msg["fire_full"], dtype=np.float32)
        if g.shape != (WORLD_W, WORLD_H):
            raise ValueError("fire_full must be 20x20")
        st.fire = (g > 0).astype(np.float32)
    else:
        for x, y in msg.get("fire_on", []):
            x, y = int(x), int(y)
            if 0 <= x < WORLD_W and 0 <= y < WORLD_H:
                st.fire[x, y] = 1.0
        for x, y in msg.get("fire_off", []):
            x, y = int(x), int(y)
            if 0 <= x < WORLD_W and 0 <= y < WORLD_H:
                st.fire[x, y] = 0.0


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


def compute_light(st: SessionState, action: int) -> Tuple[np.ndarray, int, str]:
    # if action references non-existent exit -> fall back
    if action == AUTO_ACTION or action >= len(st.exits):
        dist_target = st.dist_nearest
        eff = AUTO_ACTION
        mode = "AUTO_NEAREST"
    else:
        dist_target = st.dist_per_exit[action]
        eff = action
        mode = "GUIDE_EXIT"

    corridor = build_guidance_corridor_mask(st.layout, st.fire, st.crowd, dist_target)
    light = build_light_field(st.layout, st.fire, st.crowd, corridor)
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

                layout_msg = msg.get("layout", {})
                walls = layout_msg.get("walls", [])
                exits = layout_msg.get("exits", [])

                try:
                    layout, exit_list = parse_layout(walls, exits)
                    dist_nearest = bfs_distance_map_from_sources(layout, exit_list)
                    dist_per_exit = [bfs_distance_map_from_sources(layout, [ex]) for ex in exit_list]

                    SESSIONS[sid] = SessionState(
                        session_id=sid,
                        layout=layout,
                        exits=exit_list,
                        dist_nearest=dist_nearest,
                        dist_per_exit=dist_per_exit,
                        crowd=np.zeros((WORLD_W, WORLD_H), dtype=np.float32),
                        fire=np.zeros((WORLD_W, WORLD_H), dtype=np.float32),
                        light_prev=np.zeros((WORLD_W, WORLD_H), dtype=np.float32),
                        last_t=-1,
                    )
                    await ws.send_text(json.dumps({"type": "ack", "message": "init_ok", "session_id": sid}))
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "message": f"init_failed: {e}"}))
                continue

            if mtype == "tick":
                sid = str(msg.get("session_id", current_session_id or "default"))
                if sid not in SESSIONS:
                    await ws.send_text(json.dumps({"type": "error", "message": "No session. Send init first."}))
                    continue

                st = SESSIONS[sid]
                t = int(msg.get("t", st.last_t + 1))
                st.last_t = t

                try:
                    apply_tick_update(st, msg)
                    action = choose_action(st)
                    light, eff, mode = compute_light(st, action)
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
                        "counts": {"n_white": int(np.sum(light == -1)), "n_red": int(np.sum(light == 1))},
                    }
                    await ws.send_text(json.dumps(out))
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "message": f"tick_failed: {e}"}))
                continue

            await ws.send_text(json.dumps({"type": "error", "message": "Unknown message type."}))

    except WebSocketDisconnect:
        return


def main():
    global MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="evac_light_ppo.zip")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    try:
        MODEL = PPO.load(args.model)
        print(f"Loaded model: {args.model}")
    except Exception as e:
        MODEL = None
        print(f"Model not loaded, using AUTO fallback. Reason: {e}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()