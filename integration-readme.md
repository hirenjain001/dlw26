
Frontend (React Canvas)
    ↓ INIT / TICK
WebSocket (ws://127.0.0.1:8000/ws)
    ↓ CMD
Backend (ws_server.py)
    ↓ lights_delta
Frontend overlay renderer

{
  "type": "init",
  "session_id": "demo",
  "grid": { "w": 20, "h": 20 },
  "layout": {
    "walls": [[x,y], ...],
    "exits": [[x,y], ...]
  },
  "opts": { "max_exits": 4 }
}

{
  "type": "tick",
  "session_id": "demo",
  "t": number,
  "ts_ms": number,
  "crowd_delta": [[x,y,count], ...],
  "fire_on": [[x,y], ...],
  "fire_off": [[x,y], ...]
}

{
  "type": "cmd",
  "policy": { "action": number, "mode": "AUTO_NEAREST" },
  "lights_delta": [[x,y,"WHITE"], ...],
  "counts": { "n_white": number, "n_red": number }
}