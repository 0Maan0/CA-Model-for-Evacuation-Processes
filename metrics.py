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
import os

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

def flux(net_flow, Ntot, Ncols, Nrows, iterations):
    """
    This function determines the flux of agents of type 1 and type 2.
    Input:
    net_flow: Net forward flow of agents per iteration. (Int)
    Ntot: Total amount of agents. (Int)
    Ncol: Amount of columns in grid. (Int)
    Nrows: Amount of rows in grid. (Int)
    iterations: Number of iterations. (Int)
    Returns:
    Flux: velocity * density (Float)
    """
    velocity = net_flow / Ncols
    density = Ntot/ (Ncols * Nrows)
    return velocity * density

def congestion_index(Ntot, congested_agents):
    """	
    This function calculates the congestion index of the system.
    Input:
    Ntot: Total amount of agents. (Int)
    congested_agents: Amount of congested agents. (Int)
    Returns:
    congestion_index: The congestion index of the system per iteration. (Float)
    """
    return congested_agents / Ntot 

def plot_congestion_and_flux(agent_1_leaving, agent_2_leaving, total_fluxes_agent_1, total_fluxes_agent_2, save_path = "Figures/congestion_flux_subplot.pdf"):
    """
    This function plots the amount of agents leaving at the exits per itaand the total flux of agents in the system.
    """
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
    plt.savefig(save_path)
    plt.show()


def plot_congestion_flux(densities, results_path = "simulation_results", save_path = "Figures/total_flux.pdf"):   
    """
    This function plots the total flux of agents for varying densities (to show the critical density where congestion happens).
    """
    flux_values = np.zeros(len(densities))
    std_values = np.zeros(len(densities))
    for i, density in enumerate(densities):
        csv_path = os.path.join(results_path, f"congestion_flux_d{density}.csv")
        flux_values[i] = np.mean(np.loadtxt(csv_path, delimiter=","))
        std_values[i] = np.std(np.loadtxt(csv_path, delimiter=","))
    plt.figure(figsize=(10, 6))
    plt.scatter(densities[np.argmax(flux_values)], np.max(flux_values), color='#ef7f36', label=f'Critical density = {np.round(densities[np.argmax(flux_values)],2)}', zorder=2)
    plt.errorbar(densities, flux_values, linestyle='-', marker='o', color='#008083', zorder=1, yerr=std_values)
    plt.xlabel("Density", fontsize=14)
    plt.ylabel("Mean Flux", fontsize=14)
    plt.title("Mean flux of agents for varying density", fontsize=16)
    plt.grid(True)
    #plt.xlim([0.01, 0.45])
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_congestion_index(densities, results_path = "simulation_results", save_path = "Figures/congestion_index.pdf"):
    """
    This function plots the congestion index for varying densities (to show the critical density where congestion happens).
    """
    congestion_values = np.zeros(len(densities))
    congestion_std = np.zeros(len(densities))
    flux_values = np.zeros(len(densities))
    for i, density in enumerate(densities):
        csv_path = os.path.join(results_path, f"congestion_index_d{density}.csv")
        congestion_indexes = np.loadtxt(csv_path, delimiter=",")
        congestion_values[i] = np.mean(congestion_indexes) * 100
        congestion_std[i] = np.std(congestion_indexes) * 100
        flux_path = os.path.join(results_path, f"congestion_flux_d{density}.csv")
        flux_values[i] = np.mean(np.loadtxt(flux_path, delimiter=","))
    critical_index = np.argmax(flux_values)
    plt.figure(figsize=(10, 6))
    plt.scatter(densities[critical_index], congestion_values[critical_index], color='#ef7f36', label=f'Critical density = {np.round(densities[np.argmax(flux_values)],2)}', zorder=2)
    plt.errorbar(densities, congestion_values, linestyle='-', marker='o', color='#008083', zorder=1, yerr=congestion_std)
    plt.xlabel("Density", fontsize=14)
    plt.ylabel("CR (%)", fontsize=14)
    plt.title("Congestion Ratio for varying density", fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.xlim([0.01, 0.45])
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    densities = np.linspace(0.05, 0.43, 15)
    plot_congestion_flux(densities)
    plot_congestion_index(densities)

