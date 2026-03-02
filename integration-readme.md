# WebSocket Integration Layer (Frontend ↔ Backend)

Author: Taryn  
Branch: feature/ws-integration  
Status: Working (AUTO fallback mode active)

---

## 1. Architecture Overview

Frontend (React Canvas)
    ↓ INIT / TICK
WebSocket (ws://127.0.0.1:8000/ws)
    ↓ CMD
Backend (ws_server.py)
    ↓ lights_delta
Frontend Light Grid Store → Canvas Overlay

The frontend sends INIT once, then streams TICK updates every 200ms.
Backend replies with CMD messages containing `lights_delta`.

---

## 2. File Structure (Frontend)

simulation/frontend/src/

api/
  └── protocol.ts        # WS message types
  └── evacSocket.ts      # WebSocket client

state/
  └── lightGrid.ts       # 20x20 light grid store

components/
  └── Simulation.tsx     # Canvas + AI overlay renderer

---

## 3. WebSocket Contract

### INIT (Frontend → Backend)

Sent once when DEPLOY AI is clicked.

{
  type: "init",
  session_id: "demo",
  grid: { w: 20, h: 20 },
  layout: {
    walls: [[x,y], ...],
    exits: [[x,y], ...]
  },
  opts: { max_exits: number }
}

NOTE:
- Must include at least one exit or backend rejects with init_failed.

---

### TICK (Frontend → Backend)

Sent every 200ms if deltas exist.

{
  type: "tick",
  session_id: "demo",
  t: number,
  ts_ms: number,
  crowd_delta: [[x,y,count], ...],
  fire_on: [[x,y], ...],
  fire_off: [[x,y], ...]
}

---

### CMD (Backend → Frontend)

Backend response after processing tick.

{
  type: "cmd",
  t: number,
  ts_ms: number,
  ttl_ms: number,
  policy: { action: number, mode: "AUTO_NEAREST" | "GUIDE_EXIT" },
  lights_delta: [[x,y,"WHITE" | "RED"], ...],
  counts: { n_white: number, n_red: number }
}

Frontend applies `lights_delta` to a 20x20 grid and renders glow overlay.

---

## 4. Current Working State

✔ WebSocket connects successfully  
✔ INIT validated  
✔ TICK streaming every 200ms  
✔ Backend returns CMD  
✔ lights_delta applied correctly  
✔ Glow overlay implemented  
✔ AUTO fallback working  

---

## 5. Model Status

Current folder `evac_light_ppo/` is not a valid SB3 .zip checkpoint.

Backend logs:
"Model not loaded, using AUTO fallback"

To enable PPO mode:
- Provide proper SB3 .zip model file.
- Launch backend with:
  python3 ws_server.py --model <checkpoint_name>

---

## 6. How To Run

### Backend

cd dlw26
source .venv/bin/activate
python3 ws_server.py --model evac_light_ppo

### Frontend

cd simulation/frontend
npm install
npm run dev

---

## 7. Known Limitations

- No reconnect logic if backend dies
- No UI validation for missing exits
- No throttling on high crowd_delta frequency
- No production logging control

---

## 8. Next Improvements (Optional)

- Add model checkpoint properly
- Add reconnect + retry
- Add UI error feedback
- Add performance optimisation for overlay