Here are the exact battle plans for your Frontend Architect and your API Integrator.

If they follow these directives, they can build the entire visual simulation and the communication bridge completely independently of the RL engineer. When the RL engineer wakes up with a trained model tomorrow, you just plug it in and the system comes alive.

---

### 🔵 DIRECTIVE FOR THE FRONTEND ARCHITECT: THE DIGITAL TWIN

**Your Mission:** You are building the "Wow Factor." The judges won't look at the Python backend; they will stare at your screen. You need to build a high-performance 2D particle simulation in the browser that looks like a military-grade B2B dashboard.

#### Step 1: The Canvas Render Loop (Do not use the DOM)

If you try to render 2,000 moving people using React `<div>` tags, the browser will crash. You must use the **HTML5 `<canvas>` API** with a `requestAnimationFrame` loop.

* Draw the static walls (grey).
* Draw the exits (green).
* Draw the particles as 2-pixel cyan dots.

#### Step 2: The Physics (Boids + Phototaxis)

You don't need a heavy physics library like `Matter.js`. You just need a lightweight steering behavior script (look up "Craig Reynolds Boids algorithm in TypeScript").
Every particle needs three basic rules:

1. **Seek:** A vector pulling them toward the nearest exit coordinate.
2. **Separate:** A vector pushing them away from other particles. (Make this force exponential when they get too close—this visually creates the "Crush" bottleneck at the door).
3. **FLEE (The Phototaxis Override):** This is the most important part. If a zone's state is updated to `RED_STROBE` or `BLACK`, you apply a massive negative force vector to that area. The particles must physically violently bounce away from the AI's restricted zones.

#### Step 3: The Enterprise UI Wrapper

The canvas sits in the middle of the screen. Surround it with React components that make it look like expensive enterprise software.

* Include a live **Flow Rate Graph** (use `Recharts`).
* Add a massive red **"Crush Warning"** indicator that flashes when particles overlap too much.
* Add a toggle switch: **"Standard Evacuation" vs. "Shepherd AI Override"**. (This sets up your A/B test demo).

#### Your Deliverables for Tonight:

1. Get 1,000 dots moving from the left side of a canvas to a small gap (door) on the right side.
2. Ensure they naturally jam up and slow down at the gap.
3. Hardcode a "Red Zone" in the middle of the screen and prove the dots pathfind around it.

---

### 🟢 DIRECTIVE FOR THE API INTEGRATOR: THE TRANSLATOR

**Your Mission:** You are the central nervous system. The TS Canvas operates in continuous space (X: 1042.5, Y: 800.2). The Python RL model operates in a discrete grid (a 20x20 matrix of 1s and 0s). They cannot talk to each other without you.

#### Step 1: The WebSocket Bridge

REST APIs are too slow for a live simulation. You must establish a real-time bi-directional **WebSocket** connection.

* **Backend:** Spin up a `FastAPI` server with a WebSocket endpoint.
* **Frontend:** Use native browser `WebSocket` or `Socket.io-client` in a React `useEffect` hook.

#### Step 2: The Translation Layer (The "Basketing" Algorithm)

To save bandwidth and processing power, the frontend should not send the exact X/Y coordinates of 2,000 particles 60 times a second.

* **Frontend Task:** Write a TypeScript function that divides the canvas into a 20x20 grid. Every 500ms, count how many particles are in each grid square. Send this tiny 20x20 array (the "Density Matrix") to the backend.
* **Backend Task:** Take that 20x20 array, format it as a `numpy` tensor, and feed it to the RL Model's `model.predict(density_matrix)` function.

#### Step 3: The Mock AI (Do this before the RL is finished)

The RL guy is training the model overnight. The frontend guy needs to test his lights *now*.
You must write a **Dummy Inference Endpoint** in Python.
Write a script that receives the Density Matrix and just returns a random lighting command every 3 seconds: `{"zone_1": "RED", "zone_2": "GREEN"}`.
This allows the Frontend Architect to verify that the WebSocket works and that his particles actually flee the red lights.

#### Your Deliverables for Tonight:

1. Spin up the FastAPI WebSocket server.
2. Write the TS script to "basket" the continuous particle coordinates into a discrete 20x20 density grid.
3. Send the grid to Python, have Python print it to the terminal, and have Python send back a hardcoded dummy lighting command that the TS frontend successfully logs.

---

### The Day 2 Convergence

If your team executes this tonight, tomorrow morning looks like this:

1. The RL guy hands the API Integrator a trained `.pt` (PyTorch) or `.zip` (Stable Baselines) model file.
2. The API Integrator swaps out the "Dummy AI" for the actual `model.predict()` function.
3. You run the simulation. The TS frontend sends the live crowd density to the API. The API feeds it to the trained neural network. The neural network calculates the optimal phototaxis delay tactic. The API sends the lighting commands back to TS. The canvas renders the lights. The simulated crowd survives.

Tell them to lock in. This split guarantees nobody is bottlenecked waiting for someone else's code to compile.