import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from FFCA import FFCA
from Grid import Grid, Pos

# global constants for the FFCA
OBSTACLE = 1000
EXIT = 0
AGENT_1 = 1
AGENT_2 = 2
EMPTY = 3

# Parameters for the grid and simulation
grid_size = (20, 20)  # Grid dimensions
num_agents = 30       # Number of agents
num_steps = 50        # Number of steps in the simulation
exit_position = (0, 10)  # Exit location

# Initialize agent positions
def initialize_agents(grid_size, num_agents):
    agents = []
    while len(agents) < num_agents:
        x, y = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
        if (x, y) not in agents and (x, y) != exit_position:
            agents.append((x, y))
    return agents

# Move agents toward the exit
def move_agents(agents, exit_position, grid_size):
    new_agents = []
    for x, y in agents:
        dx = np.sign(exit_position[0] - x)
        dy = np.sign(exit_position[1] - y)

        # Randomize movement priority
        if np.random.rand() > 0.5:
            new_x, new_y = x + dx, y
        else:
            new_x, new_y = x, y + dy

        # Ensure new positions are within bounds
        new_x = max(0, min(grid_size[0] - 1, new_x))
        new_y = max(0, min(grid_size[1] - 1, new_y))

        # Add the new position if not already occupied
        if (new_x, new_y) not in new_agents and (new_x, new_y) != exit_position:
            new_agents.append((new_x, new_y))
        else:
            new_agents.append((x, y))

    return new_agents


# Visualisation of the grid and agent positions

import matplotlib.pyplot as plt
import numpy as np
import imageio

def grid_to_image(grid: Grid) -> np.ndarray:
    # Get the grid size
    rows = grid.Rmax + 2
    cols = grid.Cmax + 2
    
    # Create an empty image (matrix of zeros)
    img = np.zeros((rows, cols, 3), dtype=np.uint8)  # RGB image
    
    # Define color mapping for agents and obstacles
    color_map = {
        OBSTACLE: [0, 128, 131],  # dark green for walls
        EXIT: [142, 197, 130],  # light green for exits
        AGENT_1: [239, 127, 54],  # orange for agent 1
        AGENT_2: [41, 114, 182],  # blue for agent 2
        EMPTY: [255, 255, 255],  # white for empty cells
    }
    
    # Iterate through the grid and assign colors
    for pos, value in grid.items():
        row, col = pos.r, pos.c
        img[row, col] = color_map.get(value, [255, 255, 255])  # default to white for unknown values
    
    return img

def visualize_simulation(ffca: FFCA, steps: int, filename: str = 'simulation.gif', delay: float = 0.05):
    frames = []
    
    for i in range(steps):
        # Get image for current state of the grid
        img = grid_to_image(ffca.structure)
        
        # Append the frame to the list
        frames.append(img)
        
        # Perform a simulation step (move agents, update dynamic field, etc.)
        ffca.step()
    
    # Create and save GIF
    imageio.mimsave(filename, frames, duration=delay)
    print(f"GIF saved as {filename}")

# Call the visualization function
ffca = FFCA(20, 100, 100)  # Adjust parameters as necessary
visualize_simulation(ffca, 100)  # Create GIF for 1000 steps


