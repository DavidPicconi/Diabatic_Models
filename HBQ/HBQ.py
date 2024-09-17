"""
A module to evaluate the diabatic potential energy surface
for the HBQ molecule. 
The ground state and the excited states, as well as the diabatic couplings
between the excited states can be computed.

@author: David Picconi
"""

from Parameters import Load_Parameters
from Geometry import geoS0, masses, Omega   # Load the S0 minimum geometry
                                     # and the atomic masses

##################################

# Load the parameters
V1D, grad, Hess, c3, c4, W12, W13, W23 = Load_Parameters()

##################################
# Functions to evaluate the diabatic potentials 

def DiabaticPotential_Q(Q):
    """
    Calculate the diabatic potentials using
    the dimensionless normal modes as input.
    
    Input:
        Q  : The vector of dimensionless coordinates.
             It should have 66 dimensions
    
    Output
        W: The ground and excited state 4x4 diabatic potential matrix
           in hartrees.
           The couplings between the ground and the excited states are zero
    """
    nModes = 66
    # Check the size
    if len(Q) != nModes:
        print('ERROR: a vector of lenght 66 was expected')
        return None
    
    # Potentials
    import numpy as np
    
    W = np.zeros((4,4))
    Q1 = Q[0]
    # diagonal potentials
    for iS in range(4):
        V = V1D[iS](Q1)
        for iM in range(nModes - 1):
            V += grad[iS][iM + 1](Q1) * Q[iM + 1] \
               + 0.5 * Hess[iS][iM + 1][iM + 1](Q1) * Q[iM + 1]**2
            for jM in range(iM + 1, nModes - 1):
                V += Hess[iS][iM + 1][jM + 1](Q1) * Q[iM + 1] * Q[jM + 1]
        # cubic and quartic terms
        V += c3[iS](Q1) * Q[8]**3 + c4[iS](Q1) * Q[8]**4
        #
        W[iS,iS] = float(V)
    # diabatic couplings
    W[1,2] = W12(Q1)
    W[1,3] = W13[0](Q1) * Q[1] \
           + W13[1](Q1) * Q[2] \
           + W13[2](Q1) * Q[3] \
           + W13[3](Q1) * Q[4]
    W[2,3] = W23[0](Q1) * Q[1] \
           + W23[1](Q1) * Q[2] \
           + W23[2](Q1) * Q[3] \
           + W23[3](Q1) * Q[4]
    W[2,1] = W[1,2]
    W[3,1] = W[1,3]
    W[3,2] = W[2,3]
    
    #
    return W


    
def DiabaticPotential(geo, CheckAlignment = False):
    """
    Calculate the diabatic potentials using
    Cartesian coordinates as input.
    
    Input:
     geo : The geometry of HBQ in bohr.
             It should have shape (24,3)
     CheckAlignment : if True a message is printed to check that
                      the molecule was rotated correctly
    
    Output
        W: The ground and excited state 4x4 diabatic potential matrix
           in hartrees.
           The couplings between the ground and the excited states are zero
    """
    from Geometry import GetQCoords
    Q = GetQCoords(geo, CheckAlignment)
    #
    return DiabaticPotential_Q(Q)