import numpy as np

# Create a grid with 3 layers, 20 rows, and 20 columns (0 is wall, 1 is walkway, exit is 2)
world_x_limit = 20
world_y_limit = 20
world = np.zeros((3,world_x_limit,world_y_limit))

# Create walls around the grid
world[0, 0, :] = 1
world[0, 19, :] = 1
world[0, :, 0] = 1
world[0, :, 19]

# Create an exit 
exit_x_coord = 0
exit_y_coord = 0
world[0, exit_x_coord, exit_y_coord] = 2

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