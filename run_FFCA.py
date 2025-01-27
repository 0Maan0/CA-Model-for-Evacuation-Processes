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


def test_collision():
    # test agents
    agents = [(Pos(1, 1), 1), (Pos(1, 4), 2)]
    ffca = FFCA(2, 4, 0, agents)
    steps = 10
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        # print(ffca.structure)
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


def run():
    Ntot = 50 # total number of agents
    steps = 1000 # number of steps in the simulation
    L = 10 # number of rows
    W = 100 # number of columns
    # Calculate the mean phi for a random distribution of agents
    random_phi_values = []
    for _ in range(100):
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
    ffca = FFCA_wrap(10, 100, 100, spawn_rate=0, conflict_resolution_rate=0, verbose=False)
    steps = 10000
    ffca.show()
    for i in range(steps):
        ffca.step()
        ffca.show()


def main():
    # run()
    # debug_wrap()
    debug_wrap_2()
    pass


if __name__ == "__main__":
    main()
