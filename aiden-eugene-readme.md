Here is the exact battle plan for your Deep Learning engineer.

If they wait for the TypeScript physics engine to be finished before they start training, your team will fail. The RL model takes time to converge. They must start training **tonight**.

To do this, the RL engineer must build a **Proxy Environment**. They will train the neural network on a simplified, abstract, turn-based mathematical grid in Python. Once the AI learns the *logic* of crowd control on the grid, you export that brain via an API to control the continuous, high-fidelity physics simulation in the browser.

Here is the exact guideline to hand to your RL engineer right now.

---

### 🔴 DIRECTIVE FOR THE RL ENGINEER: THE PROXY PROTOCOL

**Your Mission:** You are not building a physics engine. You are building a purely mathematical `gymnasium` environment. You are training an agent to play a turn-based strategy game where the objective is to evacuate a 2D Numpy array as fast as possible without causing density spikes.

#### Step 1: Define the Abstract Grid (The State Space)

Do not worry about continuous X/Y coordinates or decimal points. Represent the building as a simple discrete grid (e.g., a 20x20 `numpy` matrix).

* `0` = Wall (Unwalkable)
* `1` = Empty Hallway
* `2` = The Exit
* `3` = The Hazard (Fire)
* `Negative numbers` = The Light State (e.g., `-1` for Pitch Black, `-2` for Red Strobe).

Your `observation_space` is a 3D matrix (Channels x Height x Width):

* **Layer 1:** The static building architecture (Walls/Exits).
* **Layer 2:** The current lighting state of the zones.
* **Layer 3:** The **Crowd Density Matrix**. (How many people are in each cell).

#### Step 2: Constrain the Actions (The Action Space)

**Warning:** If you let the AI control every single cell's lighting individually, the action space will be too massive, and it will never learn in 48 hours.

You must define **Control Zones** (e.g., 5 specific chokepoints or hallways).
Your `action_space` should be `Discrete(N)` or `MultiDiscrete`, where the AI chooses which specific predefined zone to plunge into "Aversion Mode" (Red Strobe/Black) and for how many 'ticks'.

#### Step 3: Fake the Physics (The Transition Dynamics)

Because you don't have the frontend's complex Boids/Social Force physics, you must write a "dumb" simulation for the Python training loop.
Use basic **Cellular Automata** logic for your `step()` function:

1. In every tick, every "person" (a counter in the density matrix) looks at its neighboring cells.
2. It moves to the cell that is closest to the exit (using a simple pre-computed distance gradient).
3. **The Crush Mechanic:** If a cell exceeds a density of $D_{max}$ (e.g., 5 people), flow rate drops to zero. Nobody can move through that cell.
4. **The Phototaxis Override:** If the RL agent changes a cell's light state to "Red Strobe", the people *refuse* to enter it, and recalculate their path.

#### Step 4: The Ruthless Reward Function

This is where the magic happens. The AI learns entirely from this math. Inside your `gymnasium` environment's `step()` function, calculate the reward ($R$):

$$R = (\alpha \times \text{Evacuated\_This\_Step}) - (\beta \times \text{Crush\_Penalty})$$

* **Evacuated_This_Step:** +10 points for every person that hits the 'Exit' cell.
* **Crush_Penalty:** -50 points for every cell where `density > 5`.
* *The Result:* The AI will initially try to rush everyone to the door, hit the crush penalty, and fail. Over 10,000 episodes, it will realize that using its action space to temporarily block hallways with "Red Strobes" keeps the density below 5, maximizing its score.

#### Step 5: The Algorithm (`stable-baselines3`)

Do not write the neural net from scratch in PyTorch. Import `stable-baselines3`.
Use **PPO (Proximal Policy Optimization)**. It is the most stable and requires the least hyperparameter tuning for grid-based environments.

```python
from stable_baselines3 import PPO
from your_custom_env import ShepherdEnv

env = ShepherdEnv()
# Use a CNN policy because your observation space is a 2D spatial grid
model = PPO("CnnPolicy", env, verbose=1) 
model.learn(total_timesteps=500000)
model.save("shepherd_agent")

```

#### Step 6: The API Handoff (Day 2)

While you are training this model, the frontend guy is building the beautiful, continuous TS canvas.
On Day 2, you build a FastAPI endpoint: `POST /get_action`.
The frontend sends you its current crowd density state. You map that continuous state to your discrete grid, feed it to `model.predict(obs)`, and return the lighting command: `{"zone_3": "RED_STROBE"}`. The frontend visually executes it.

---

### Your Deliverables for Tonight (In Order):

1. Hardcode a 20x20 `numpy` array representing a simplified version of the building the frontend team is building.
2. Write the `ShepherdEnv(gym.Env)` class with a basic cellular automata `step()` function.
3. Verify the "dumb" crowd naturally jams at the exit when no lights are used.
4. Start the PPO training loop. Let it run while you sleep.

Tell your RL engineer to focus *exclusively* on the grid and the reward math. The visual flair is not their problem. If the math works on the grid, it will work on the canvas.