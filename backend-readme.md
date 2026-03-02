# Smart Evacuation Lighting – Backend (RL + WebSocket)

This backend provides a reinforcement learning (RL) inference service that dynamically controls building lights to guide people toward exits and away from hazards during emergencies.

The system receives live crowd density and fire updates from the frontend and returns per-cell lighting commands (**WHITE** / **RED** / **OFF**).

## Architecture Overview

```text
Frontend (simulation / sensors)
        ↓
  Crowd + Fire state (20×20 grid)
        ↓
Backend (RL inference)
        ↓
  Lighting commands (per-cell)
```

**Important:**
* The backend **does NOT** simulate crowd movement or fire during deployment.
* It only performs real-time decision making based on incoming state.

## Project Structure

### 1. evac_core.py
Core logic and utilities shared by training and inference.
Includes:
* Grid constants (`WORLD_W`, `WORLD_H`)
* BFS distance computation to exits
* Guidance corridor generation
* Light field construction
    * **WHITE** → guidance
    * **RED** → danger (fire / overcrowding)
* Efficient delta encoding: `light_grid_to_delta(old_light, new_light)`

### 2. train_env.py
Training environment for the RL model.
This file:
* Simulates crowd movement and fire spread
* Defines reward:
    * `+` evacuation
    * `-` deaths
    * `-` overcrowding
    * `-` slow evacuation
* Trains PPO policy

Run training:
```bash
python train_env.py --timesteps 200000 --model evac_light_ppo.zip
```

Output:
`evac_light_ppo.zip`
*(After training, this file is not used during runtime.)*

### 3. ws_server.py
Real-time inference server (FastAPI + WebSocket).

Responsibilities:
* Accept frontend sessions
* Maintain session state: layout, exits, crowd grid, fire grid, previous light grid
* Run: `action = model.predict(obs)`
* Compute lighting
* Return per-cell delta updates

Start server:
```bash
python ws_server.py --model evac_light_ppo.zip
```

Server runs at:
* HTTP: `http://127.0.0.1:8000`
* WebSocket: `ws://127.0.0.1:8000/ws`

Health check:
`GET /`

### 4. contract_test_ws.py
Integration / contract test.
This verifies:
* WebSocket connectivity
* JSON schema correctness
* Lighting response validity

Run (in a second terminal while server is running):
```bash
python contract_test_ws.py
```

Expected output:
```text
INIT OK
TICK 0 OK
...
CONTRACT TEST PASSED
```
*Use this if frontend integration fails.*

## Frontend ↔ Backend Protocol

### INIT (once per session)
```json
{
  "type": "init",
  "session_id": "demo",
  "grid": {"w": 20, "h": 20},
  "layout": {
    "walls": [[x, y], ...],
    "exits": [[x, y], ...]
  },
  "opts": {
    "max_exits": 3
  }
}
```

Response:
```json
{
  "type": "ack",
  "message": "init_ok",
  "session_id": "demo"
}
```

### TICK (send continuously)
Recommended rate: 5–10 Hz
```json
{
  "type": "tick",
  "session_id": "demo",
  "t": 184,
  "ts_ms": 1700000000000,
  "crowd_delta": [[x, y, count], ...],
  "fire_on": [[x, y], ...],
  "fire_off": [[x, y], ...]
}
```
*Only send changed cells (sparse updates).*

### CMD (server response)
```json
{
  "type": "cmd",
  "t": 184,
  "ts_ms": 1700000000123,
  "ttl_ms": 500,
  "policy": {
    "action": 1,
    "mode": "GUIDE_EXIT"
  },
  "lights_delta": [
    [x, y, "WHITE"],
    [x, y, "RED"],
    [x, y, "OFF"]
  ],
  "counts": {
    "n_white": 42,
    "n_red": 16
  }
}
```
Frontend should:
* Update only the cells in `lights_delta`
* Keep previous light state for all other cells

## Grid Convention
* Size: 20 × 20
* Coordinates: `[x, y]`
* Frontend may simulate continuous positions, but must bucket into grid cells before sending.

## Running the System
Terminal 1:
```bash
python ws_server.py --model evac_light_ppo.zip
```

Terminal 2:
```bash
python contract_test_ws.py
```

Then connect frontend to:
`ws://127.0.0.1:8000/ws`

## Fallback Behavior
If the model fails to load:
* Server automatically switches to `AUTO_NEAREST`
* People are guided toward nearest exit using distance only
* System remains functional

## Design Goals
* Real-time inference (<10 ms)
* Sparse network traffic
* Supports: multiple exits, arbitrary layouts, dynamic fire, variable crowd density
* Backend maintains lightweight session state only

## Hackathon Workflow
1. Train model (`train_env.py`)
2. Start server (`ws_server.py`)
3. Verify with contract test
4. Connect frontend

*Backend is now ready for live integration.*