# Project 'Moth' - Smart Evacuation Lighting (React + RL + WebSocket)

This project provides a complete digital twin for emergency evacuation. It features a continuous React physics simulation (Frontend) and a reinforcement learning (RL) inference service (Backend) that dynamically controls building lights to guide people toward exits and away from hazards.

The system receives live crowd density and fire updates from the frontend and returns per-cell lighting commands (**WHITE** / **RED** / **OFF**).

## Architecture Overview

```text
Frontend (continuous simulation / React canvas)
        ↓
Frontend Link (converts simulation data to WebSocket payload)
        ↓
  Crowd + Fire state (sparse 20×20 grid delta)
        ↓
Backend (RL inference server)
        ↓
  Lighting commands (per-cell)
        ↓
Frontend (renders Potential Field / Swarm follows lights)
```

**Important:**
* The backend **does NOT** simulate crowd movement or fire during deployment.
* It only performs real-time decision making based on incoming state.
* The frontend handles all continuous physics (boids separation, wall sliding, obstacle repulsion) and translates the visual canvas into the discrete grid.

---

## Project Structure

### Backend (Python)

#### 1. evac_core.py
Core logic and utilities shared by training and inference.
Includes:
* Grid constants (`WORLD_W`, `WORLD_H`)
* BFS distance computation to exits
* Guidance corridor generation
* Light field construction
    * **WHITE** → guidance
    * **RED** → danger (fire / overcrowding)
* Efficient delta encoding: `light_grid_to_delta(old_light, new_light)`

#### 2. train_env.py
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
*(After training, the `.zip` file is used during runtime.)*

#### 3. ws_server.py
Real-time inference server (FastAPI + WebSocket).

Responsibilities:
* Accept frontend sessions
* Maintain session state: layout, exits, crowd grid, fire grid, previous light grid
* Run: `action = model.predict(obs)`
* Compute lighting
* Return per-cell delta updates

#### 4. contract_test_ws.py
Integration / contract test to verify WebSocket connectivity, JSON schema correctness, and lighting response validity.

---

### Frontend (React / TypeScript / Vite)

The frontend is a custom-built, continuous 60fps 2D physics engine serving as the visual testing ground and interactive dashboard for the digital twin. 

#### 1. Simulation.tsx (The Digital Twin Dashboard)
* **Interactive Tooling:** Features a God-Mode UI allowing users to dynamically draw `Walls`, `Exits`, `Fires`, and targeted `Spawn Zones` directly onto the HTML5 Canvas.
* **Pitch Scenarios:** Includes pre-loaded architectural challenges (e.g., *Obstacle Course*, *Office Maze*, and even a *Sandbox Mode*) that dynamically scale to the screen size for one-click live demonstrations.
* **Performance:** Utilizes `useRef` and direct DOM manipulation to maintain 60fps rendering without triggering React re-renders.

#### 2. Particle.ts (The Swarm Brain & Physics)
* **Potential Field Navigation:** Particles act as receivers for the AI's light grid. They are gravitationally pulled by **WHITE** cells and violently repelled by **RED** cells.
* **Ice-Wall Collision:** Implements advanced physical sliding and vector repulsion, allowing the crowd to flow smoothly around hard wall boundaries without getting stuck in "friction traps."
* **Crowd Crush Prevention:** Uses boids-style separation forces to ensure humans maintain personal space and push away from each other during bottlenecks.

#### 3. Link.ts (The State Parser)
* **The Basketing Algorithm:** Translates the continuous floating-point X/Y canvas into a discrete 20x20 math matrix every 200ms.
* **Sparse Delta Encoding:** Compares the current frame's matrix against the previous frame, generating lightweight JSON arrays of *only what changed* (`crowd_delta`, `fire_on`, `fire_off`) to ensure sub-10ms network latency.

---

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

### TICK (send continuously)
Recommended rate: 5–10 Hz. *Only send changed cells (sparse updates).*
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

### CMD (server response)
Frontend updates *only* the cells in `lights_delta` and keeps previous light states for all other cells.
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
  ]
}
```

---

## Running the System

### 1. Start the Backend Server
Open your first terminal and boot the FastAPI WebSocket server:
```bash
python ws_server.py --model evac_light_ppo.zip
```
*(Server runs at HTTP `http://127.0.0.1:8000` and WebSocket `ws://127.0.0.1:8000/ws`)*

### 2. Verify Backend Contract (Optional)
In a second terminal, verify the Python socket is receiving payloads:
```bash
python contract_test_ws.py
```

### 3. Start the Frontend Digital Twin
Open a third terminal, navigate to your frontend directory, and boot the Vite server:
```bash
npm install
npm run dev
```

### 4. Run the Live Integration
1. Open `http://localhost:5173` in your browser.
2. Select a pre-built architectural layout from the bottom toolbar (e.g., **S2: Fire Reroute**).
3. Click **[+] SPAWN ZONE**, click, and drag a rectangle on the canvas to drop a crowd of humans.
4. Click **DEPLOY AI** to establish the WebSocket connection.
5. Watch the AI instantly map the optimal **WHITE** escape routes and actively update **RED** danger zones if you draw dynamic fires.

## Fallback Behavior
If the model fails to load, the server automatically switches to `AUTO_NEAREST`. People are guided toward the nearest exit using distance only, ensuring the system remains functional.

## Setup

### 1. Create a virtual environment
Run the following command in your terminal to create a local environment named `env`:
```bash
python -m venv env
```

### 2. Activate the environment
Activation commands vary depending on your Operating System and the shell you are using:

* **Windows (Command Prompt):**
    ```cmd
    .\env\Scripts\activate
    ```
* **Windows (PowerShell):**
    ```powershell
    .\env\Scripts\Activate.ps1
    ```
* **macOS / Linux (bash/zsh):**
    ```bash
    source env/bin/activate
    ```

> **Note for Windows Users:** If you receive an execution policy error in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Session` to allow script execution for the current session.

### 3. Install dependencies
Once the environment is active (you should see `(env)` in your terminal prompt), install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Deactivate
When you are finished working, you can exit the virtual environment by simply typing:
```bash
deactivate
```

---

## Troubleshooting

* **Python Command:** If `python` isn't recognized, try using `python3` instead.
* **Pip Upgrade:** It is often helpful to ensure `pip` is up to date inside the environment:
    ```bash
    python -m pip install --upgrade pip
    ```
* **Missing Requirements:** If `requirements.txt` is missing, ensure you are in the root directory of the project.
