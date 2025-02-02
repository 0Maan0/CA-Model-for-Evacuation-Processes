# CA-Model-for-Evacuation-Processes
CA Model for Evacuation Processes for the course Complex System Simulation

This repository contains a **Floor Field Cellular Automaton (FFCA)** simulation to analyze pedestrian **lane formation** and **congestion** nunder different conditions. The model evaluates various parameters like density, horizontal bias, and conflict resolution rate to understand emergent crowd behavior.

## Project Structure

- **FFCA_wrap.py** contains the main simulation function to run the simulation to determine the ciritcal density at which congestion happens. Initializes agents, simulates their movement, and handles conflict resolution. Provides the FFCA_wrap class that interacts with the grid and agents.
- **Grid.py** defines the grid structure and positions of agents. The Grid class manages the grid layout, while the Pos class represents individual positions within it.
- **metrics.py** contains the functions to plot and compute the flux, order parameer and congetsion rate.
- **Vizualitation.py** contains the function use to create a gif of the simulation process.
- **run_FFCA.py** imports the FFCA class and runs the simulation./
- **run_simulation.py** imports the FFCA class and runs some standard scenarios to see the visually see the FFCA interactions.
- **Data_Analysis.py** contains the code to run the plopts for data analysis 
