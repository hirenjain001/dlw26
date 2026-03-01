import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Create a grid with 3 layers, 20 rows, and 20 columns (0 is wall, 1 is walkway, exit is 2)
world_x_limit = 20
world_y_limit = 20
world = np.zeros((3,world_x_limit,world_y_limit))

# Create walls around the grid
world[0, 0, :] = 1
world[0, 19, :] = 1
world[0, :, 0] = 1
world[0, :, 19] = 1

# Create an exit 
exit_x_coord = 1
exit_y_coord = 1
world[0, exit_x_coord, exit_y_coord] = 2

# This just randomly creates walls in the room (PLEASE DELETE THIS AFTER HIREN IMPLEMENTS THE PHYSICS BASED THING)
def generate_random_obstacles(world, num_obstacles = 5):
    world[0, 1:19, 1:19] = 0
    count = 0
    while count < num_obstacles:
        start_x = np.random.randint(2,15)
        start_y = np.random.randint(2,15)

        obs_width = np.random.randint(2,5)
        obs_length = np.random.randint(2,5)

        world[0, start_x:start_x+obs_width, start_y:start_y+obs_length] = 1
        count += 1

        world[0, exit_x_coord, exit_y_coord] = 2
    return world
    
world = generate_random_obstacles(world, num_obstacles=5)

# Create a dist_map for the people to reference
dist_map = np.full((20,20),999)
dist_map[exit_x_coord, exit_y_coord] = 0
queue = [(exit_x_coord, exit_y_coord)]

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

while queue:
    coords = queue.pop(0)
    for hor_add, ver_add in directions:
        update_coords = (coords[0] + hor_add, coords[1] + ver_add)
        if update_coords[0] >= world_x_limit or update_coords[0] < 0 or update_coords[1] >= world_y_limit or update_coords[1] < 0:
            continue
        if world[0, update_coords[0], update_coords[1]] == 1:
            continue
        if dist_map[update_coords[0], update_coords[1]] != 999:
            continue
        dist_map[update_coords[0],update_coords[1]] = dist_map[coords[0], coords[1]] + 1
        queue.append(update_coords)

# Create the movement tech
def move_crowd(world, dist_map):
    new_crowd_layer = np.zeros((20,20))
    light_penalty = 100
    evacuated_this_turn = 0
    for x in range(world_x_limit):
        for y in range(world_y_limit):     
            if world[2,x,y] > 0:
                score_list=[]
                for hor_add, ver_add in directions:
                    if x + hor_add >= world_x_limit or x + hor_add < 0 or y + ver_add>= world_y_limit or y + ver_add < 0 or world[0,x+hor_add,y+ver_add] == 1   :
                        score_list.append(9999)
                        continue
                    score = dist_map[x+hor_add,y+ver_add] + (world[1,x+hor_add,y+ver_add]*light_penalty)
                    # Add extra functions that affect scoring here :D
                    score_list.append(score)
                
                lowest_score_index = np.argmin(score_list)
                if score_list[lowest_score_index] < 9999:
                    optimal_direction_x, optimal_direction_y = directions[lowest_score_index]
                    if world[0, x+optimal_direction_x, y+optimal_direction_y] == 2:
                        evacuated_this_turn += world[2, x, y]
                    else:
                        new_crowd_layer[x+optimal_direction_x,y+optimal_direction_y] += world[2,x,y]
                else:
                    new_crowd_layer[x,y] += world[2,x,y]
    return new_crowd_layer, evacuated_this_turn

# This is just to plot and see if my building layout (layer 0) and dist map actually is working
# # 1. Create a masked array where we hide the 999 values
# # This allows the color scale to focus ONLY on the path distances (0 to ~30)
# masked_dist_map = np.ma.masked_where(dist_map >= 999, dist_map)

# plt.figure(figsize=(12, 5))

# # Plot Layer 0 (Building Layout)
# plt.subplot(1, 2, 1)
# plt.title("Building Layout (Walls/Exit)")
# plt.imshow(world[0], cmap='gray') # Gray makes walls look like concrete

# # Plot the Gradient GPS
# plt.subplot(1, 2, 2)
# plt.title("Crowd GPS (Blue = Near Exit, Red = Far)")

# # Use 'nipy_spectral' or 'jet' for high contrast
# im = plt.imshow(masked_dist_map, cmap='nipy_spectral', interpolation='none')
# plt.colorbar(im, label='Steps to Exit')

# plt.tight_layout()
# plt.show()


# Create the AI action
# Zoning Region
zones = {
    0: (0, 10, 0, 10), # 1st Quadrant
    1: (0, 10, 10, 20), # 2nd Quadrant
    2: (10, 20, 0, 10), # 3rd Quadrant
    3: (10, 20, 10, 20), # 4th Quadrant
    4: None
}

# Actual AI Action Code (Please edit this, it is quite basic now)
def apply_action(world, action_index):
    world[1, :, :] = 0
    zone = zones.get(action_index)
    if zone is not None:
        x_left, x_right, y_low, y_high = zone
        world[1, x_left:x_right, y_low:y_high] = 1
    
    return world

def fire_spread(world, spread_chance = 0.1):
    fire_coords = np.argwhere(world[0] == 3)
    new_fire_grid = world[0].copy()

    for curr_fire_x, curr_fire_y in fire_coords:
        for dir_x, dir_y in directions:
            fire_spread_x = curr_fire_x + dir_x
            fire_spread_y = curr_fire_y + dir_y
            if 0 <= fire_spread_x < 20 and 0 <= fire_spread_y < 20 and world[0, fire_spread_x, fire_spread_y] == 0:
                if np.random.random() < spread_chance:
                    new_fire_grid[fire_spread_x, fire_spread_y] = 3

    world[0] = new_fire_grid
    return world

class ShepherdEnv(gym.Env):
    def __init__(self, world_layout, dist_map):
        super(ShepherdEnv, self).__init__()
        self.initial_layout = world_layout.copy()
        self.world = world_layout.copy()
        self.dist_map = dist_map
        self.max_steps = 200
        self.current_step = 0

        self.action_space = gym.spaces.Discrete(5)
        
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (3, 20, 20) , dtype = np.float32)

    def step(self, action):
        self.world = apply_action(self.world, action)
        self.world[2], evacuated = move_crowd(self.world, self.dist_map)
        self.world = fire_spread(self.world, spread_chance=0.1)
        self.current_step += 1

        burned_people = np.sum(self.world[2][self.world[0]==3])

        reward = evacuated*20
        reward -= (burned_people * 50)
        reward -= 1

        if np.any(self.world[2] > 5):
            reward -=50
        
        terminated = np.sum(self.world[2]) == 0
        truncated = self.current_step >= self.max_steps

        if truncated and not terminated:
            reward -= 500

        return self.world.astype(np.float32), reward, terminated, truncated, {}

    # Reset the question for the model type shii
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.world[0] = self.initial_layout[0].copy()
        self.world[1, :, :] = 0
        self.world[2, :, :] = 0
        self.current_step = 0

        # this is the random function to add people into the situation for the model to play around with
        people_placed = 0
        while people_placed < 50:
            rand_ppl_placement_x = np.random.randint(0, world_x_limit)
            rand_ppl_placement_y = np.random.randint(0, world_y_limit)

            if self.world[0, rand_ppl_placement_x, rand_ppl_placement_y] == 0:
                self.world[2, rand_ppl_placement_x, rand_ppl_placement_y] += 1
                people_placed += 1
        
        # this is the random fire starting point (you guys can tweak it to have multiple fire starting points in a simulation if you want)
        fire_started = False
        while not fire_started:
            fire_x, fire_y = np.random.randint(1,19), np.random.randint(1,19)
            if self.world[0, fire_x, fire_y] == 0 and self.world[2, fire_x, fire_y] == 0:
                self.world[0, fire_x, fire_y] = 3
                fire_started = True

        return self.world.astype(np.float32), {}
    
# Time to train!
env = ShepherdEnv(world, dist_map)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003, 
    tensorboard_log="./ppo_shepherd_logs/" 
)
print("Training Started...")
model.learn(total_timesteps = 500000)
model.save("shepherd_ai_model")
print("Model saved successfully!")
