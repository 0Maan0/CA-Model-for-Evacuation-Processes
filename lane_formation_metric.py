import numpy as np
"""
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.75.051402
"""
# Order parameter to determine the lane formation in the system
def order_parameter(Ntot, N1, N2):    
    '''
    Ntot: Total number of particles
    N1: Number of particles in each row with direction 1 (Array)
    N2: Number of particles in each row with direction 2 (Array)
    '''
    filled_rows = 0
    phi = 0
    for row in range(len(N1)):
        if N1[row] + N2[row] > 0:
            phi += (((N1[row] - N2[row]))/ (N1[row] + N2[row])) **2
            filled_rows += 1 
    if Ntot == 0 or filled_rows == 0:
        print("No particles in the system, order parameter is not defined.")
        return np.nan
    return phi/filled_rows

if __name__ == "__main__":
    Ntot = 50
    N1 = np.array([0, 0])
    N2 = np.array([0, 0])
    phi_value = order_parameter(Ntot, N1, N2)
    print(f"""Order parameter: {phi_value}""")