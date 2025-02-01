import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from FFCA_wrap import FFCA_wrap
from Grid import Grid, Pos
import imageio
import cv2

# global constants for the FFCA
OBSTACLE = 1000
EXIT = 0
AGENT_1 = 1
AGENT_2 = 2
EMPTY = 3

# Visualisation of the grid and agent positions

def grid_to_image(grid: Grid, scale: int = 20) -> np.ndarray:
    rows = grid.Rmax + 2
    cols = grid.Cmax + 2
    img = np.zeros((rows, cols, 3), dtype=np.uint8)

    color_map = {
        OBSTACLE: [0, 128, 131],
        EXIT: [142, 197, 130],
        AGENT_1: [239, 127, 54],
        AGENT_2: [41, 114, 182],
        EMPTY: [255, 255, 255],
    }

    for pos, value in grid.items():
        row, col = pos.r, pos.c
        img[row, col] = color_map.get(value, [255, 255, 255])

    # Resize the image using nearest neighbor interpolation to keep sharp edges
    img = cv2.resize(img, (cols * scale, rows * scale), interpolation=cv2.INTER_NEAREST)
    return img

def visualize_simulation(ffca: FFCA_wrap, steps: int, filename: str = 'simulation3.gif', delay: float = 0.05):
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
    ffca = FFCA_wrap(12, 20, 10, spawn_rate=0.025,
                    conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                    static_field_strength=2.5, dynamic_field_strength=3,
                    horizontal_bias=5000)  
    visualize_simulation(ffca, 1000)  # Create GIF for 1000 steps


