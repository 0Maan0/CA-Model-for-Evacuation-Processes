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

if __name__ == "__main__":
    # Open phi_values.csv file and plot the order parameter
    phi_values = np.loadtxt("phi_values.csv", delimiter=",")
    iterations = len(phi_values)
    plot_order_parameter(phi_values, iterations)