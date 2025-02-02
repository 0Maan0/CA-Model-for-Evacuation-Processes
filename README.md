# CA-Model-for-Evacuation-Processes
CA Model for Evacuation Processes for the course Complex System Simulation

This repository contains a **Floor Field Cellular Automaton (FFCA)** simulation to analyze pedestrian **lane formation** and **congestion** nunder different conditions. The model evaluates various parameters like density, horizontal bias, and conflict resolution rate to understand emergent crowd behavior.

## Project Structure

FFCA_wrap contains the main simulation function to run the simulation to determine the ciritcal density at which congestion happens: run run_simulation.py
metrics.py contains the functions to plot and compute the flux, order parameer and congetsion rate.
Vizualitation.py contains the function use to create a gif of the simulation process. 
