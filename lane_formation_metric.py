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
    return (np.sum( ((N1 - N2)/(N1 + N2)) **2) ) / Ntot

Ntot = 250
N1 = np.array([50, 50, 1, 1, 1])
N2 = np.array([0, 0, 49, 49, 49])
phi = order_parameter(Ntot, N1, N2)
print(f"""Order parameter: {phi}""")