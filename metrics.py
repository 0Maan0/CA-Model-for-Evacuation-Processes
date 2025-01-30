"""
University      University of Amsterdam
Course:         Complex systems simulation
Authors:        Pjotr Piet, Maan Scipio, Sofia Tete
IDs:            12714933, 15899039, 15830608
Description:    This file contains functions for calculating the order parameter and flux, 
                along with some functions the plot these parameters to evaluate them in terms 
                of lane formation and congestion.

https://journals.aps.org/pre/abstract/10.1103/PhysRevE.75.051402
"""
import numpy as np
import matplotlib.pyplot as plt

# Order parameter to determine the lane formation in the system
def order_parameter(Ntot, N1, N2):    
    '''
    This function calculates the order parameter for the system.
    Input:
    Ntot: Total number of particles
    N1: Number of particles in each row with direction 1 (Array)
    N2: Number of particles in each row with direction 2 (Array)
    Returns:
    phi: Order parameter (Float)
    '''
    filled_rows = 0
    phi = 0
    for row in range(len(N1)):
        if N1[row] + N2[row] > 0:
            phi += ((N1[row] - N2[row])/ (N1[row] + N2[row])) **2
            filled_rows += 1 
    if Ntot == 0 or filled_rows == 0:
        print("No particles in the system, order parameter is not defined.")
        return np.nan
    return phi/filled_rows

def mean_order_parameter(phi, phi0):
    '''
    This function calculates the mean order parameter for the system.
    Input:
    phi: Order parameter (Float)
    phi0: Mean order parameter for random distribution (Float)
    Returns:
    phi bar: Mean Order Parameter (Float)
    '''
    return (phi - phi0) / (1 - phi0)

def agent_flow(agent_1_leaving, agent_2_leaving, agent_1_entering, agent_2_entering):
    '''
    This function calculates the flow of agents in the system.
    Input: 
    agent_1_leaving: Number of agents of type 1 leaving the system (Int)
    agent_2_leaving: Number of agents of type 2 leaving the system (Int)
    agent_1_entering: Number of agents of type 1 entering the system (Int)
    agent_2_entering: Number of agents of type 2 entering the system (Int)
    Returns:
    total_flux_agent_1: Net flow of agents of type 1 through the exits. (Int)
    total_flux_agent_2: Net flow of agents of type 2 through the exits. (Int)
    '''
    total_flux_agent_1 = agent_1_entering - agent_1_leaving
    total_flux_agent_2 = agent_2_entering - agent_2_leaving

    return total_flux_agent_1, total_flux_agent_2

def congestion_metric(agent_1_leaving, agent_2_leaving, Ntot, Ncol, Nrows, iterations):
    """
    This function determines the flux of agents of type 1 and type 2.
    Input:
    agent_1_leaving: Amount of agents of type 1 leaving each row. (Array)
    agent_2_leaving Amount of agents of type 2 leaving each row. (Array)
    Ntot: Tota amount of agents. (Int)
    Ncol: Amount of columns in grid. (Int)
    Nrows: Amount of rows in grid. (Int)
    iterations: Number of iterations. (Int)
    Returns:
    Flux: velocity * density (Float)
    """
    total1 = np.sum(agent_1_leaving)
    total2 = np.sum(agent_2_leaving)
    total_normalised_flow = ( total1 + total2 ) / Ntot
    velocity = total_normalised_flow * Ncol
    density = Ntot / (Ncol * Nrows)
    return velocity * density 

def plot_congestion_and_flux(agent_1_leaving, agent_2_leaving, total_fluxes_agent_1, total_fluxes_agent_2):
    iterations = len(agent_1_leaving)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Number of Agents Leaving
    ax[0].plot(range(iterations), agent_1_leaving, linestyle='-', color='#ef7f36', label='Agent 1')
    ax[0].plot(range(iterations), agent_2_leaving, linestyle='-', color='#008083', label='Agent 2')
    ax[0].set_ylabel("Number of Agents Leaving", fontsize=14)
    ax[0].legend(fontsize = 13)
    ax[0].grid(True)

    # Total Flux
    ax[1].plot(range(iterations), total_fluxes_agent_1, linestyle='-', color='#ef7f36')
    ax[1].plot(range(iterations), total_fluxes_agent_2, linestyle='-', color='#008083')
    ax[1].set_xlabel("Iterations", fontsize=14)
    ax[1].set_ylabel("Total Flux of agents", fontsize=14)

    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("Figures/congestion_flux_subplot.pdf")
    plt.show()


def plot_congestion_flux(densities):
    flux_values = np.zeros(len(densities))
    for i, density in enumerate(densities):
        flux_values[i] = np.mean(np.loadtxt(f"simulation_results/congestion_flux_d{density}.csv", delimiter=","))
    plt.figure(figsize=(10, 6))
    plt.plot(densities, flux_values, linestyle='-')
    plt.xlabel("Density", fontsize=14)
    plt.ylabel("Total Flux", fontsize=14)
    plt.title("Total Flux vs Iterations", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("Figures/total_flux.pdf")
    plt.show()

if __name__ == "__main__":
    
    densities = np.linspace(0.04, 0.25, 10)
    plot_congestion_flux(densities)

    # # Open flux_values_1.csv and flux_values_2.csv files and plot the total flux
    flux_values_1 = np.loadtxt("simulation_results/flux_values_1.csv", delimiter=",")
    flux_values_2 = np.loadtxt("simulation_results/flux_values_2.csv", delimiter=",")
    # #plot_total_flux(flux_values_1, flux_values_2)
    agent1 = np.loadtxt("simulation_results/agent_1_leaving.csv", delimiter=",")
    agent2 = np.loadtxt("simulation_results/agent_2_leaving.csv", delimiter=",")
    # detect_congestion(agent1, agent2)
    plot_congestion_and_flux(agent1, agent2, flux_values_1, flux_values_2)
    
