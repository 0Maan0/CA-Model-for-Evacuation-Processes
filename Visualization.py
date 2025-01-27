import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from FFCA_wrap import FFCA_wrap
from Grid import Grid, Pos

# global constants for the FFCA
OBSTACLE = 1000
EXIT = 0
AGENT_1 = 1
AGENT_2 = 2
EMPTY = 3

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

def visualize_simulation(ffca: FFCA_wrap, steps: int, filename: str = 'r30_c50_n100_kd3_hori_bias_5000.gif', delay: float = 0.05):
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

if __name__ == "__main__":
    # Call the visualization function
    ffca = FFCA_wrap(30, 50, 100, spawn_rate=0.025,
                    conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                    static_field_strength=2.5, dynamic_field_strength=3,
                    horizontal_bias=5000)  
    visualize_simulation(ffca, 2000)  # Create GIF for 1000 steps


