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
from metrics import order_parameter, mean_order_parameter, plot_order_parameter, agent_flow, congestion_metric
import time
import numpy as np
from Visualization import visualize_simulation, grid_to_image
import imageio
import cv2

def run_density_comparison(cmax, rmax, steps, densities):
    densities = np.linspace(0.04, 0.25, 10)
    cmax = 50
    rmax = 25
    steps = 1000
    for density in densities:
        Ntot = int(density * cmax * rmax)
        ffca = FFCA_wrap(rmax, cmax, Ntot, spawn_rate=0.025,
                        conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                        static_field_strength=2.5, dynamic_field_strength=3,
                        horizontal_bias=5000)  
        random_phi_values = []
        for _ in range(500):
            ffca = FFCA_wrap(rmax, cmax, Ntot, spawn_rate=0.025,
                        conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                        static_field_strength=2.5, dynamic_field_strength=3,
                        horizontal_bias=5000)  
            N1, N2 = ffca.agents_in_row(ffca.structure)
            random_phi_values.append(order_parameter(Ntot, N1, N2, agents = True))
        phi_zero = np.mean(random_phi_values)
        ffca = FFCA_wrap(rmax, cmax, Ntot, spawn_rate=0.025,
                        conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                        static_field_strength=2.5, dynamic_field_strength=3,
                        horizontal_bias=5000)  
        #visualize_simulation(ffca, 1000)
        phi_values = np.zeros(steps)
        flux_values_1 = np.zeros(steps)
        flux_values_2 = np.zeros(steps)
        total_flux_counter_1 = 0
        total_flux_counter_2 = 0
        agent_1_leaving = np.zeros(steps)
        agent_2_leaving = np.zeros(steps)
        leave_count1 = 0
        leave_count2 = 0


        congestion_flux = np.zeros(steps)
        frames = []
        for i in range(steps):
            # Get image for current state of the grid
            img = grid_to_image(ffca.structure)
            img_large = cv2.resize(img, (img.shape[1] * 10, img.shape[0] * 10), interpolation=cv2.INTER_NEAREST)
            # Append the frame to the list
            frames.append(img_large)
            current_phi = order_parameter(Ntot, *ffca.agents_in_row(ffca.structure), agents = True)
            current_mean_phi = mean_order_parameter(current_phi, phi_zero)
            agent_fluxes = agent_flow(*ffca.agents_at_exit(ffca.structure))
            total_flux_counter_1 += agent_fluxes[0]
            total_flux_counter_2 += agent_fluxes[1]
            flux_values_1[i] = total_flux_counter_1
            flux_values_2[i] = total_flux_counter_2

            leave_count1 += agent_fluxes[2]
            leave_count2 += agent_fluxes[3]
            agent_1_leaving[i] = leave_count1
            agent_2_leaving[i] = leave_count2
            phi_values[i] = current_mean_phi

            congestion_flux[i] = congestion_metric(agent_fluxes[2], agent_fluxes[3], Ntot, cmax, rmax, steps)
            ffca.step()
        filename = f"Figures/gif{density}.gif"
        imageio.mimsave(filename, frames)
        print(f"GIF saved as {filename}")
        # save 2 arrays in csv file
        np.savetxt(f"simulation_results/congestion_flux_d{density}.csv", congestion_flux, delimiter = ",")
        np.savetxt(f"simulation_results/flux_values_1_d{density}.csv", flux_values_1, delimiter=",")
        np.savetxt(f"simulation_results/flux_values_2_d{density}.csv", flux_values_2, delimiter=",")
        np.savetxt(f"simulation_results/agent_1_leaving_d{density}.csv", agent_1_leaving, delimiter=",")
        np.savetxt(f"simulation_results/agent_2_leaving_d{density}.csv", agent_2_leaving, delimiter=",")
        # save phi_values in csv file
        np.savetxt(f"simulation_results/phi_values_d{density}.csv", phi_values, delimiter=",")




def main():
    run_all()
    pass


if __name__ == "__main__":
    main()