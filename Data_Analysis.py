from FFCA import FFCA
from Grid import Grid, Pos
from metrics import order_parameter, mean_order_parameter, plot_order_parameter
import time
import numpy as np
import matplotlib.pyplot as plt

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

# DIfferent horizontal bias
# Different density values