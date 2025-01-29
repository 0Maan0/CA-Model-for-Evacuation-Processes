"""
University      University of Amstedam
Course:         Complex systems simulation
Authors:        Pjotr Piet, Maan Scipio, Sofia Tete
IDs:            12714933, 15899039, 15830608
Description:    This file imports the FFCA class and runs some standard
scenarios to see the visually see the FFCA interactions.
"""

# from FFCA_wrap import FFCA
from FFCA import FFCA
from FFCA_wrap import FFCA_wrap
from FFCA_wrap import print_grid
from Grid import Grid, Pos
from metrics import order_parameter, mean_order_parameter, plot_order_parameter, agent_flux, congestion_metric
import time
import numpy as np
from Visualization import visualize_simulation, grid_to_image
import imageio
import cv2

def test_collision():
    # test agents
    agents = [(Pos(1, 1), 1), (Pos(1, 4), 2)]
    ffca = FFCA(2, 4, 0, agents)
    steps = 10
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()


def test_small():
    cmax = 40
    rmax = 3
    r = 2
    c = cmax
    left_agents = [(Pos(r, 1 + 2 * i), 1) for i in range(1)]
    right_agents = [(Pos(r, c - 2 * i), 2) for i in range(10)]
    agents = left_agents + right_agents

    ffca = FFCA(rmax, cmax, 0, agents)
    steps = 300
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()


def test_big():
    ffca = FFCA(10, 100, 20, spawn_rate=0.06, conflict_resolution_rate=0,
                dynamic_field_strength=10)
    steps = 1000
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()


def run_wrap():
    cmax = 40
    rmax = 3
    r = 2
    c = cmax
    left_agents = [(Pos(r, 1 + 2 * i), 1) for i in range(1)]
    right_agents = [(Pos(r, c - 2 * i), 2) for i in range(10)]
    agents = left_agents + right_agents

    ffca = FFCA_wrap(rmax, cmax, 0, agents)
    steps = 300
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()

def run_wrap_statistics():
    cmax = 50
    rmax = 20
    r = 2
    c = cmax
    left_agents = [(Pos(r, 1 + 2 * i), 1) for i in range(1)]
    right_agents = [(Pos(r, c - 2 * i), 2) for i in range(10)]
    agents = left_agents + right_agents
    Ntot = 50
    dfs = 3
    ffca = FFCA_wrap(rmax, cmax, Ntot, dynamic_field_strength=dfs)
    steps = 1000
    ffca.show()
    random_phi_values = []
    for _ in range(100):
        ffca = FFCA_wrap(rmax, cmax, Ntot, dynamic_field_strength=dfs)
        N1, N2 = ffca.agents_in_row(ffca.structure)
        random_phi_values.append(order_parameter(Ntot, N1, N2))
    phi_zero = np.mean(random_phi_values)
    ffca = FFCA_wrap(rmax, cmax, Ntot, dynamic_field_strength=dfs)
    phi_values = np.zeros(steps)
    flux_values_1 = np.zeros(steps)
    flux_values_2 = np.zeros(steps)
    total_flux_counter_1 = 0
    total_flux_counter_2 = 0
    agent_1_leaving = np.zeros(steps)
    agent_2_leaving = np.zeros(steps)
    leave_count1 = 0
    leave_count2 = 0
    for i in range(steps):
        time.sleep(0.05)
        #print(f"Step {i}")
        current_phi = order_parameter(Ntot, *ffca.agents_in_row(ffca.structure))
        current_mean_phi = mean_order_parameter(current_phi, phi_zero)
        #ffca.show()
        agent_fluxes = agent_flux(*ffca.agents_at_exit(ffca.structure))
        total_flux_counter_1 += agent_fluxes[0]
        total_flux_counter_2 += agent_fluxes[1]
        flux_values_1[i] = total_flux_counter_1
        flux_values_2[i] = total_flux_counter_2

        leave_count1 += agent_fluxes[2]
        leave_count2 += agent_fluxes[3]
        agent_1_leaving[i] = leave_count1
        agent_2_leaving[i] = leave_count2
        phi_values[i] = current_mean_phi
        ffca.step()
        if agent_fluxes[2] == 0 or agent_fluxes[3] == 0:
            print(f"Step {i}")
            ffca.show()
    # save 2 arrays in csv file
    np.savetxt(f"simulation_results/flux_values_1_dfs{dfs}.csv", flux_values_1, delimiter=",")
    np.savetxt(f"simulation_results/flux_values_2_dfs{dfs}.csv", flux_values_2, delimiter=",")
    np.savetxt(f"simulation_results/agent_1_leaving_dfs{dfs}.csv", agent_1_leaving, delimiter=",")
    np.savetxt(f"simulation_results/agent_2_leaving_dfs{dfs}.csv", agent_2_leaving, delimiter=",")
    # save phi_values in csv file
    np.savetxt(f"simulation_results/phi_values_dfs{dfs}.csv", phi_values, delimiter=",")
#plot_order_parameter(phi_values, steps)

def run():
    Ntot = 50 # total number of agents
    steps = 1000 # number of steps in the simulation
    L = 20 # number of rows
    W = 50 # number of columns
    # Calculate the mean phi for a random distribution of agents
    random_phi_values = []
    for _ in range(500):
        ffca = FFCA(L, W, Ntot)
        N1, N2 = ffca.agents_in_row(ffca.structure)
        random_phi_values.append(order_parameter(Ntot, N1, N2))
    phi_zero = np.mean(random_phi_values)

    ffca = FFCA(L, W, Ntot)
    phi_values = np.zeros(steps)
    flux_values_1 = np.zeros(steps)
    flux_values_2 = np.zeros(steps)
    total_flux_counter_1 = 0
    total_flux_counter_2 = 0
    frames = []
    for i in range(steps):
        # Get image for current state of the grid
        img = grid_to_image(ffca.structure)
        
        # Append the frame to the list
        frames.append(img)
        time.sleep(0.05)
        # print(f"Step {i}")
        current_phi = order_parameter(Ntot, *ffca.agents_in_row(ffca.structure))
        current_mean_phi = mean_order_parameter(current_phi, phi_zero)
        ffca.show()
        agent_fluxes = agent_flux(*ffca.agents_at_exit(ffca.structure))
        total_flux_counter_1 += agent_fluxes[0]
        total_flux_counter_2 += agent_fluxes[1]
        flux_values_1[i] = total_flux_counter_1
        flux_values_2[i] = total_flux_counter_2
        phi_values[i] = current_mean_phi
        ffca.step()
    imageio.mimsave(filename, frames, duration=delay)
    print(f"GIF saved as {filename}")
    # save 2 arrays in csv file
    np.savetxt("flux_values_1.csv", flux_values_1, delimiter=",")
    np.savetxt("flux_values_2.csv", flux_values_2, delimiter=",")
    # save phi_values in csv file
    np.savetxt("phi_values.csv", phi_values, delimiter=",")
    #plot_order_parameter(phi_values, steps)


def debug_wrap():
    # diagonal crash seems to go correct though
    # left_agents = [(Pos(2, 4), 2)]
    # right_agents = [(Pos(2, 1), 1), (Pos(1, 2), 1)]

    # wrap around:
    l = [(Pos(1, 2), 2)]
    r = [(Pos(1, 3), 1)]

    agents = l + r

    ffca = FFCA_wrap(1, 4, 0, agents, verbose=False)
    steps = 5
    print_grid(ffca.structure)
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        print_grid(ffca.structure)


# use a bigger random grid to find the error situation
def debug_wrap_2():
    ffca = FFCA_wrap(30, 30, 100, spawn_rate=0, conflict_resolution_rate=0)
    steps = 10000
    ffca.show()
    for i in range(steps):
        ffca.step()
        ffca.show()

def run_fun():
    # generate one big collumn
    rows = 30
    cols = 30
    y1 = 4
    y2 = 27
    l = [(Pos(r, c), 1) for r in range(1, rows + 1) for c in range(y1 - 2, y1 + 1, 2)]
    r = [(Pos(r, c), 2) for r in range(1, rows + 1) for c in range(y2 - 2, y2 + 1, 2)]
    agents = l + r
    ffca = FFCA_wrap(rows, cols, 0, agents)
    steps = 100
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()

def run_all():
    densities = np.linspace(0.04, 0.25, 10)
    cmax = 50
    rmax = 25
    for density in densities:
        Ntot = int(density * cmax * rmax)
        # y1 = 4
        # y2 = 27
        # l = [(Pos(r, c), 1) for r in range(1, rmax + 1) for c in range(y1 - 2, y1 + 1, 2)]
        # r = [(Pos(r, c), 2) for r in range(1, rmax + 1) for c in range(y2 - 2, y2 + 1, 2)]
        #agents = l + r
        ffca = FFCA_wrap(rmax, cmax, Ntot, spawn_rate=0.025,
                        conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                        static_field_strength=2.5, dynamic_field_strength=3,
                        horizontal_bias=5000)  

        steps = 1000
        random_phi_values = []
        for _ in range(100):
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
            #time.sleep(0.05)
            #print(f"Step {i}")
            current_phi = order_parameter(Ntot, *ffca.agents_in_row(ffca.structure), agents = True)
            #print(current_phi, phi_zero)
            current_mean_phi = mean_order_parameter(current_phi, phi_zero)
            agent_fluxes = agent_flux(*ffca.agents_at_exit(ffca.structure))
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
    #  run()
    # run_wrap()
    # run_wrap_statistics()
    # run()
    # test_collision()
    # test_small()
    # run()
    # debug_wrap()
    # debug_wrap_2()
    # run_fun()
    run_all()
    pass


if __name__ == "__main__":
    main()



#critical density for congestion in for example traffic and look at the critical point and compare?  
#cluster size or distribution of flux powerlaw
#why certain density parameter ==> relate to existing theoretical construct and explain differences 