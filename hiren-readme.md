You are the showstopper. If the RL engineer is the brain and the API guy is the nervous system, you are the face of the entire operation. The judges will make their decision based entirely on whether your simulation *looks and feels* like real, terrifying crowd physics.

Because you are working in TypeScript, you have to be incredibly careful. If you try to manage the state of 2,000 moving particles using React's `useState`, the browser will drop to 2 frames per second and crash. You must bypass React's render cycle for the physics.

Here is your exact blueprint for building the continuous TypeScript simulation.

---

### 🟡 DIRECTIVE FOR THE PHYSICS SIMULATOR: THE ENGINE

#### Step 1: The Canvas Architecture (Bypassing React)

You will use React to build the UI dashboard around the simulation, but the simulation itself must live inside an HTML5 `<canvas>` element referenced via a `useRef` hook.

1. Create a `Simulation.tsx` component.
2. Hook into a `requestAnimationFrame` loop. This allows the browser to natively update the canvas 60 times a second without triggering React re-renders.
3. Every frame, you clear the canvas (`ctx.clearRect`), update the math for every particle, and redraw the dots (`ctx.fillRect`).

#### Step 2: The Particle Class (The Boids)

Do not overcomplicate the physics. Every human is just a dot with three properties: `position`, `velocity`, and `acceleration`.

Create a `Particle` class in TypeScript. Inside the update loop, every particle calculates its next move based on three forces:

* **The Goal Force (Egress):** A vector pointing straight from the particle's current X/Y to the nearest Exit X/Y.
* **The Separation Force (The Crush):** This is crucial. Loop through nearby particles. If the distance between two particles is less than 3 pixels, apply a massive opposing force. *This mathematically causes the bottleneck at the door.*
* **The Wall Repulsion:** If a particle gets too close to a coordinate marked as a "Wall", bounce it away so they don't walk through solid objects.

#### Step 3: The Phototaxis Override (The Hack)

This is how you integrate with the RL model without waiting for the API guy.
Your particles want to go to the exit. You need to write a function that temporarily overrides that desire if a light flashes.

Write an `applyPhototaxis(lightZones)` method in your `Particle` class:

* The canvas will have predefined "Zones" (e.g., Hallway A, Hallway B).
* If the RL agent sets Hallway A to `RED_STROBE`, and a particle's X/Y is inside or near Hallway A, you multiply their base velocity by `-5` and reverse their heading.
* Visually, the moment the API changes the zone state, the particles will violently scatter away from that hallway like bugs fleeing a flashlight, recalculating a path down a longer, unlit hallway.

#### Step 4: The A/B Visuals

You need to visually prove the AI is working.

1. Map the walls in stark grey.
2. Draw the particles in a bright cyan.
3. When they get crushed (when their Separation Force spikes), change their color to **bright red**. This visually screams "DANGER" to the judges.
4. When the RL agent activates a dark zone, draw a semi-transparent black square over that canvas area. When it activates the runway, pulse green squares. Make the intervention obvious.

---

### Your Deliverables for Tonight (In Order):

1. **The Box:** Get a React component rendering an 800x600 canvas.
2. **The Flow:** Spawn 500 cyan dots on the left side of the screen and make them calculate a vector to move to a 20-pixel "door" on the right side.
3. **The Jam:** Implement the Separation Force. Watch the 500 dots clump up at the 20-pixel door, slowing down the flow rate. Make them turn red when they touch.
4. **The Flee:** Hardcode a "Red Zone" box in the middle of the screen. Write the Phototaxis logic to make the dots actively route around that box.

### Implementation Commits

```text
chore: initialize ts react canvas component with requestAnimationFrame loop
feat(physics): create particle class with position, velocity, and acceleration vectors
feat(physics): implement separation force logic to simulate crowd crush density
feat(routing): add goal-seeking vector to pull particles toward egress coordinates
feat(phototaxis): implement aversion logic to aggressively repulse particles from active red zones
feat(render): add dynamic color state to particles turning red when collision density exceeds threshold

```

This is the exact sequence you need to code tonight. Do not worry about the Python AI or the WebSockets yet. Build the fish tank, make the fish flock, and give them a reason to run.

Are you clear on how to write the `requestAnimationFrame` loop in React without causing memory leaks, or do you want the boilerplate for that component?