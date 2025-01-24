"""
University      University of Amstedam
Course:         Complex systems simulation
Authors:        Pjotr Piet, Maan Scipio, Sofia Tete
IDs:            12714933, 15899039, 15830608
Description:    This file imports the FFCA class and runs some standard
scenarios to see the visually see the FFCA interactions.
"""

from FFCA import FFCA
from Grid import Grid, Pos
from lane_formation_metric import order_parameter, mean_order_parameter, plot_order_parameter
import time
import numpy as np


def test_collision():
    # test agents
    agents = [(Pos(1, 1), 1), (Pos(1, 5), 2)]
    ffca = FFCA(1, 5, 0, agents)
    steps = 10
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()


# test_collision()


def test_small():
    cmax = 40
    rmax = 3
    r = 2
    c = cmax
    left_agents = [(Pos(r, 1 + 2 * i), 1) for i in range(2)]
    right_agents = [(Pos(r, c - 2 * i), 2) for i in range(10)]
    agents = left_agents + right_agents

    ffca = FFCA(rmax, cmax, 0, agents)
    steps = 30
    ffca.show()
    for i in range(steps):
        time.sleep(0.5)
        ffca.step()
        ffca.show()


test_small()


def test_big():
    ffca = FFCA(10, 100, 20, spawn_rate=0.06, conflict_resolution_rate=0,
                dynamic_field_strength=10)
    steps = 1000
    ffca.show()
    for i in range(steps):
        ffca.step()
        ffca.show()


# test_big()


def run():
    Ntot = 50 # total number of agents
    steps = 1000 # number of steps in the simulation
    # Calculate the mean phi for a random distribution of agents
    random_phi_values = []
    for _ in range(500):
        ffca = FFCA(10, 100, Ntot)
        N1, N2 = ffca.agents_in_row(ffca.structure)
        random_phi_values.append(order_parameter(Ntot, N1, N2))
    phi_zero = np.mean(random_phi_values)

    ffca = FFCA(10, 100, Ntot)
    phi_values = np.zeros(steps)

    for i in range(steps):
        time.sleep(0.05)
        # print(f"Step {i}")
        current_phi = order_parameter(Ntot, *ffca.agents_in_row(ffca.structure))
        current_mean_phi = mean_order_parameter(current_phi, phi_zero)
        phi_values[i] = current_mean_phi
        ffca.step()
        ffca.show()

    # save phi_values in csv file
    np.savetxt("phi_values.csv", phi_values, delimiter=",")
    plot_order_parameter(phi_values, steps)


# run()

