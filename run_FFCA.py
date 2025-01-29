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
from metrics import order_parameter, mean_order_parameter, plot_order_parameter, agent_flux
import time
import numpy as np


def run_wrap_statistics():
    cmax = 30
    rmax = 30
    r = 2
    c = cmax
    left_agents = [(Pos(r, 1 + 2 * i), 1) for i in range(1)]
    right_agents = [(Pos(r, c - 2 * i), 2) for i in range(10)]
    agents = left_agents + right_agents
    Ntot = 100
    dfs = 5
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
    steps = 2000 # number of steps in the simulation
    L = 30 # number of rows
    W = 30 # number of columns
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
    for i in range(steps):
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
        ffca.show()
    # save 2 arrays in csv file
    np.savetxt("flux_values_1.csv", flux_values_1, delimiter=",")
    np.savetxt("flux_values_2.csv", flux_values_2, delimiter=",")
    # save phi_values in csv file
    np.savetxt("phi_values.csv", phi_values, delimiter=",")
    #plot_order_parameter(phi_values, steps)


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


def run_wrap():
    ffca = FFCA_wrap(25, 50, 25, spawn_rate=0, conflict_resolution_rate=0, horizontal_bias=5000,
                     dynamic_field_strength=3)
    steps = 1000
    ffca.show()
    for i in range(steps):
        ffca.step()
        ffca.show()


def main():
    # run_wrap_statistics()
    # run()
    run_wrap()
    # run_fun()
    pass


if __name__ == "__main__":
    main()
