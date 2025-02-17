# CA-Model-for-Evacuation-Processes
CA Model for Evacuation Processes for the course Complex System Simulation

University: University of Amsterdam  
Course: Complex Systems Simulation  
Authors: Pjotr Piet, Maan Scipio, Sofia Tété 
IDs: 12714933, 15899039, 15830608  


This repository contains a **Floor Field Cellular Automaton (FFCA)** simulation to analyze pedestrian **lane formation** and **congestion** nunder different conditions. The model evaluates various parameters like density, horizontal bias, and conflict resolution rate to understand emergent crowd behavior.

## Project Structure

- **FFCA_wrap.py** contains the main simulation function to run the simulation to determine the ciritcal density at which congestion happens. Initializes agents, simulates their movement, and handles conflict resolution. Provides the FFCA_wrap class that interacts with the grid and agents.
- **Grid.py** defines the grid structure and positions of agents. The Grid class manages the grid layout, while the Pos class represents individual positions within it.
- **metrics.py** contains the functions to plot and compute the flux, order parameer and congetsion rate.
- **Vizualitation.py** contains the function use to create a gif of the simulation process.
- **run_FFCA.py** imports the FFCA class and runs the simulation.
- **run_simulation.py** imports the FFCA class and runs some standard scenarios to see the visually see the FFCA interactions.
- **Data_Analysis.py** contains the code to run the plopts for data analysis.
- **requirements.txt** contains the required dependencies for this project.

## Folders 
- **test/** contains the test scripts (test_FFCA_wrap.py, test_grid.py, test_pos.py) and a script to run them all (run_tests.sh).
- **Presentation/** contains the powerpoint used for the final presentation.
- **gifs/**, **simulation_results/**, **Figures/** and  **data_analysis/** contain results from the simulations.

## Requirements

To run this project, the following Python libraries are required:

- `numpy`: For numerical computations and array handling.
- `imageio`: For saving and generating GIFs.
- `opencv-python`: For image processing tasks.
- `matplotlib`: For visualizations and plotting metrics.
- `pandas`: For handling CSV files and performing data analysis.
- `cv2`: For additional image resizing and grid visualization if needed.

To install the required dependencies, run the following command in your terminal: `pip install -r requirements.txt`.
