import numpy as np
import matplotlib.pyplot as plt
"""
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.75.051402
"""
# Order parameter to determine the lane formation in the system
def order_parameter(Ntot, N1, N2, agents = False):    
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
    if agents == False:
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
    '''
    return (phi - phi0) / (1 - phi0)

def plot_order_parameter(phi_values):
    '''
    This function plots the order parameter vs iterations.
    Input:
    phi_values: List of order parameters (List)
    phi0: Mean order parameter for random distribution (Float)
    '''
    # Make plot pretty
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(phi_values)), phi_values, linestyle='-', color='tab:blue', label='Order Parameter')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Order Parameter", fontsize=14)
    plt.title("Order Parameter vs Iterations", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("order_parameter.pdf")
    plt.show()
    return None

def agent_flux(agent_1_leaving, agent_2_leaving, agent_1_entering, agent_2_entering):
    '''
    This function calculates the flux of agents in the system.
    Input: 
    agent_1_leaving: Row names where agents of type 1 are leaving the system (Int)
    agent_2_leaving: Row names where agents of type 2 are leaving the system (Int)
    agent_1_entering: Row names where agents of type 1 are entering the system (Int)
    agent_2_entering: Row names where agents of type 2 are entering the system (Int)
    '''
    nr_rows = len(agent_1_leaving)
    row_flux_agent_1 = np.zeros(nr_rows)
    row_flux_agent_2 = np.zeros(nr_rows)

    for i in range(nr_rows):
        row_flux_agent_1[i] = agent_1_entering[i] - agent_1_leaving[i]
        row_flux_agent_2[i] = agent_2_entering[i] - agent_2_leaving[i]

    total_flux_agent_1 = np.sum(row_flux_agent_1)
    total_flux_agent_2 = np.sum(row_flux_agent_2)

    amount_agent_1_leaving = np.sum(agent_1_leaving)
    amount_agent_2_leaving = np.sum(agent_2_leaving)

    return total_flux_agent_1, total_flux_agent_2, amount_agent_1_leaving, amount_agent_2_leaving

def detect_congestion(agent_1_leaving, agent_2_leaving):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(agent_1_leaving)), agent_1_leaving, linestyle='-', color='#ef7f36', label='Agent 1 Leaving')
    plt.plot(range(len(agent_2_leaving)), agent_2_leaving, linestyle='-', color='#008083', label='Agent 2 Leaving')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Number of Agents Leaving", fontsize=14)
    plt.legend()
    plt.show()
    plt.savefig("Figures/congestion.pdf")

def plot_total_flux(total_fluxes_agent_1, total_fluxes_agent_2):
    #plot of how the flux changes over time at each iteration
    iterations = len(total_fluxes_agent_1)
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), total_fluxes_agent_1, linestyle='-', color='#ef7f36', label='Total Flux Agent 1')
    plt.plot(range(iterations), total_fluxes_agent_2, linestyle='-', color='#008083', label='Total Flux Agent 2')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Total Flux", fontsize=14)
    plt.title("Total Flux vs Iterations", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("total_flux.pdf")
    plt.show()
    return None    

import matplotlib.pyplot as plt

def plot_congestion_and_flux(agent_1_leaving, agent_2_leaving, total_fluxes_agent_1, total_fluxes_agent_2):
    iterations = len(agent_1_leaving)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # First subplot - Number of Agents Leaving
    ax[0].plot(range(iterations), agent_1_leaving, linestyle='-', color='#ef7f36', label='Agent 1')
    ax[0].plot(range(iterations), agent_2_leaving, linestyle='-', color='#008083', label='Agent 2')
    ax[0].set_ylabel("Number of Agents Leaving", fontsize=14)
    ax[0].legend(fontsize = 13)
    ax[0].grid(True)

    # Second subplot - Total Flux
    ax[1].plot(range(iterations), total_fluxes_agent_1, linestyle='-', color='#ef7f36')
    ax[1].plot(range(iterations), total_fluxes_agent_2, linestyle='-', color='#008083')
    ax[1].set_xlabel("Iterations", fontsize=14)
    ax[1].set_ylabel("Total Flux of agents", fontsize=14)

    #ax[1].legend()
    ax[1].grid(True)

    # Adjust layout for clarity
    plt.tight_layout()

    # Save the figure
    plt.savefig("Figures/congestion_flux_subplot.pdf")

    # Show the plot
    plt.show()

# Example usage (provide real data in place of placeholders)
# plot_congestion_and_flux(agent_1_leaving, agent_2_leaving, total_fluxes_agent_1, total_fluxes_agent_2)

def congestion_metric(agent_1_leaving, agent_2_leaving, Ntot, Ncol, Nrows, iterations):
    total1 = np.sum(agent_1_leaving)
    total2 = np.sum(agent_2_leaving)
    total_normalised_flow = ( total1 + total2 ) / Ntot
    velocity = total_normalised_flow * Ncol
    density = Ntot / (Ncol * Nrows)
    return velocity * density


def plot_congestion_flux(densities):
    #print(np.sum(fluxes)/len(fluxes))
    #iterations = len(fluxes)
    #colors = ['b', 'r', 'cyan', 'pink', 'black']
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
    plt.savefig("total_flux.pdf")
    plt.show()

if __name__ == "__main__":
    
    densities = np.linspace(0.04, 0.25, 10)
    plot_congestion_flux(densities)

    # #Open phi_values.csv file and plot the order parameter
    # phi_values = np.loadtxt("simulation_results/phi_values_dfs5.csv", delimiter=",")
    # iterations = len(phi_values)
    # plot_order_parameter(phi_values)
    # # Open flux_values_1.csv and flux_values_2.csv files and plot the total flux
    flux_values_1 = np.loadtxt("simulation_results/flux_values_1.csv", delimiter=",")
    flux_values_2 = np.loadtxt("simulation_results/flux_values_2.csv", delimiter=",")
    # #plot_total_flux(flux_values_1, flux_values_2)
    agent1 = np.loadtxt("simulation_results/agent_1_leaving.csv", delimiter=",")
    agent2 = np.loadtxt("simulation_results/agent_2_leaving.csv", delimiter=",")
    # detect_congestion(agent1, agent2)
    plot_congestion_and_flux(agent1, agent2, flux_values_1, flux_values_2)
    
