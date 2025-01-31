"""
University      University of Amstedam
Course:         Complex systems simulation
Authors:        Pjotr Piet, Maan Scipio, Sofia Tete
IDs:            12714933, 15899039, 15830608
Description:    This file imports the FFCA class and runs some standard
scenarios to see the visually see the FFCA interactions.
"""
from FFCA_wrap import FFCA_wrap
from FFCA_wrap import print_grid
from Grid import Grid, Pos
from metrics import order_parameter, mean_order_parameter, agent_flow, flux, congestion_index
import time
import numpy as np
from Visualization import visualize_simulation, grid_to_image
import imageio
import cv2

def run_density_comparison(cmax, rmax, steps, densities):
    for density in densities:
        Ntot_half = int(density * cmax * rmax//2)
        Ntot = Ntot_half * 2
        ffca = FFCA_wrap(rmax, cmax, Ntot_half, spawn_rate=0.025,
                        conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                        static_field_strength=2.5, dynamic_field_strength=3,
                        horizontal_bias=5000)

        random_phi_values = []
        for _ in range(100):
            ffca = FFCA_wrap(rmax, cmax, Ntot_half, spawn_rate=0.025,
                        conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                        static_field_strength=2.5, dynamic_field_strength=3,
                        horizontal_bias=5000)
            N1, N2 = ffca.agents_in_row(ffca.structure)
            random_phi_values.append(order_parameter(Ntot, N1, N2))
        phi_zero = np.mean(random_phi_values)
        ffca = FFCA_wrap(rmax, cmax, Ntot_half, spawn_rate=0.025,
                        conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                        static_field_strength=2.5, dynamic_field_strength=3,
                        horizontal_bias=5000)
        phi_values = np.zeros(steps)
        congestion_indexes = np.zeros(steps)
        congestion_fluxes = np.zeros(steps)

        frames = []
        for i in range(steps):
            img = grid_to_image(ffca.structure)
            frames.append(img)
            current_phi = order_parameter(Ntot_half, *ffca.agents_in_row(ffca.structure))
            current_mean_phi = mean_order_parameter(current_phi, phi_zero)
            ffca.step()
            agent_fluxes = agent_flow(*ffca.agents_at_exit())

            phi_values[i] = current_mean_phi
            congested_agents = Ntot - ffca.get_amount_agents_moved_forward()
            congestion_indexes[i] = congestion_index(Ntot, congested_agents)
            net_flow = ffca.global_movement()
            congestion_fluxes[i] = flux(net_flow, Ntot, cmax, rmax, steps)
        filename = f"gifs/gif{density}.gif"
        imageio.mimsave(filename, frames)
        print(f"GIF saved as {filename}")
        # save 2 arrays in csv file
        np.savetxt(f"simulation_results/congestion_flux_d{density}.csv", congestion_fluxes, delimiter = ",")
        np.savetxt(f"simulation_results/congestion_index_d{density}.csv", congestion_indexes, delimiter=",")
        # save phi_values in csv file
        np.savetxt(f"simulation_results/phi_values_d{density}.csv", phi_values, delimiter=",")




def main():
    densities = np.linspace(0.05, 0.43, 15)
    cmax = 50
    rmax = 25
    steps = 1000
    run_density_comparison(cmax, rmax, steps, densities)
    pass


if __name__ == "__main__":
    main()