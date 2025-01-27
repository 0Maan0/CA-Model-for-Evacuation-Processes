import numpy as np
import matplotlib.pyplot as plt
"""
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.75.051402
"""
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
    '''
    return (phi - phi0) / (1 - phi0)

def plot_order_parameter(phi_values, iterations):
    '''
    This function plots the order parameter vs iterations.
    Input:
    phi_values: List of order parameters (List)
    phi0: Mean order parameter for random distribution (Float)
    '''
    # Make plot pretty
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), phi_values, linestyle='-', color='tab:blue', label='Order Parameter')
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
    #print(f"Number of rows: {nr_rows}")
    #print(f"row_flux_agent_1: {row_flux_agent_1}")
    #print(f"row_flux_agent_2: {row_flux_agent_2}")

    for i in range(nr_rows):
        row_flux_agent_1[i] = agent_1_entering[i] - agent_1_leaving[i]
        row_flux_agent_2[i] = agent_2_entering[i] - agent_2_leaving[i]

    total_flux_agent_1 = np.sum(row_flux_agent_1)
    total_flux_agent_2 = np.sum(row_flux_agent_2)
    #print(f"Total flux of agent 1: {total_flux_agent_1}")
    #print(f"Total flux of agent 2: {total_flux_agent_2}")

    return total_flux_agent_1, total_flux_agent_2

def plot_total_flux(total_fluxes_agent_1, total_fluxes_agent_2):
    #plot of how the flux changes over time at each iteration
    iterations = len(total_fluxes_agent_1)
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), total_fluxes_agent_1, linestyle='-', color='tab:blue', label='Total Flux Agent 1')
    plt.plot(range(iterations), total_fluxes_agent_2, linestyle='-', color='tab:red', label='Total Flux Agent 2')
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

def plot_dynamic_field_strength():
    phi_dsf3 = np.loadtxt("simulation_results/phi_values_dfs3.0.csv", delimiter=",")
    phi_dsf5 = np.loadtxt("simulation_results/phi_values_dfs5.0.csv", delimiter=",")
    phi_dsf7 = np.loadtxt("simulation_results/phi_values_dfs7.0.csv", delimiter=",")
    phi_dsf9 = np.loadtxt("simulation_results/phi_values_dfs9.0.csv", delimiter=",")
    iterations = range(len(phi_dsf3))
    print(phi_dsf3)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, phi_dsf3, linestyle='-', color='tab:blue', label='Dynamic Field Strength 3')
    plt.plot(iterations, phi_dsf5, linestyle='-', color='tab:red', label='Dynamic Field Strength 5')
    plt.plot(iterations, phi_dsf7, linestyle='-', color='tab:green', label='Dynamic Field Strength 7')
    plt.plot(iterations, phi_dsf9, linestyle='-', color='tab:orange', label='Dynamic Field Strength 9')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Order Parameter", fontsize=14)
    plt.title("Order Parameter vs Iterations", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.savefig("Figures/order_parameter_dynamic_field_strength.pdf")
    plt.show()

    flux_values_1_dfs3 = np.loadtxt("simulation_results/flux_values_1_dfs3.0.csv", delimiter=",")
    flux_values_1_dfs5 = np.loadtxt("simulation_results/flux_values_1_dfs5.0.csv", delimiter=",")
    flux_values_1_dfs7 = np.loadtxt("simulation_results/flux_values_1_dfs7.0.csv", delimiter=",")
    flux_values_1_dfs9 = np.loadtxt("simulation_results/flux_values_1_dfs9.0.csv", delimiter=",")
    flux_values_2_dfs3 = np.loadtxt("simulation_results/flux_values_2_dfs3.0.csv", delimiter=",")
    flux_values_2_dfs5 = np.loadtxt("simulation_results/flux_values_2_dfs5.0.csv", delimiter=",")
    flux_values_2_dfs7 = np.loadtxt("simulation_results/flux_values_2_dfs7.0.csv", delimiter=",")
    flux_values_2_dfs9 = np.loadtxt("simulation_results/flux_values_2_dfs9.0.csv", delimiter=",")
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, flux_values_1_dfs3, linestyle='-', color='tab:blue', label='Dynamic Field Strength 3, agent 1')
    plt.plot(iterations, flux_values_1_dfs5, linestyle='-', color='tab:red', label='Dynamic Field Strength 5, agent 1')
    plt.plot(iterations, flux_values_1_dfs7, linestyle='-', color='tab:green', label='Dynamic Field Strength 7, agent 1')
    plt.plot(iterations, flux_values_1_dfs9, linestyle='-', color='tab:orange', label='Dynamic Field Strength 9, agent 1')
    plt.plot(iterations, flux_values_2_dfs3, linestyle='--', color='tab:pink', label='Dynamic Field Strength 3, agent 2')
    plt.plot(iterations, flux_values_2_dfs5, linestyle='--', color='tab:brown', label='Dynamic Field Strength 5, agent 2')
    plt.plot(iterations, flux_values_2_dfs7, linestyle='--', color='tab:purple', label='Dynamic Field Strength 7, agent 2')
    plt.plot(iterations, flux_values_2_dfs9, linestyle='--', color='tab:cyan', label='Dynamic Field Strength 9, agent 2')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Flux", fontsize=14)
    plt.title("Flux vs Iterations", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.savefig("Figures/flux_dynamic_field_strength.pdf")
    plt.show()

    

if __name__ == "__main__":
    #Open phi_values.csv file and plot the order parameter
    #phi_values = np.loadtxt("Figures/phi_values.csv", delimiter=",")
    #iterations = len(phi_values)
    #plot_order_parameter(phi_values, iterations)
    # Open flux_values_1.csv and flux_values_2.csv files and plot the total flux
    #flux_values_1 = np.loadtxt("flux_values_1.csv", delimiter=",")
    #flux_values_2 = np.loadtxt("flux_values_2.csv", delimiter=",")
    #plot_total_flux(flux_values_1, flux_values_2)
    plot_dynamic_field_strength()