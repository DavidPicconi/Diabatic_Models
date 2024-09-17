"""
Module to read and manipulate the geometry

@author: David Picconi
"""

import numpy as np
###############################

# Read the S0 minimum geometry
mass = {'H': 1.00782504, 'C': 12.0, 'N': 14.003074, 'O': 15.99491}
with open('geo_S0min.dat') as geoS0File:
    geom   = geoS0File.readlines()
    geoS0  = np.array([[float(x) for x in l.split()[1:]] for l in geom])
    masses = np.array([mass[l.split()[0]] for l in geom])
#    
del geom, mass 

nAtoms = len(geoS0)

# Align to the center of mass
RCM = (masses @ geoS0) / np.sum(masses)
geoS0 = geoS0 - RCM
geoS0 = geoS0.reshape(nAtoms * 3) * 1.88973
del RCM

# Read the normal modes, their frequencies and the effective modes

### U0, w0, Ueff, TR
U0   = np.loadtxt('U0.dat')   # normal modes
Ueff = np.loadtxt('Ueff.dat') # effective modes
w0   = np.loadtxt('w0.dat')   # S0 frequencies
TR   = np.loadtxt('TR.dat')   # translations/rotations
Omega = Ueff.T @ np.diag(w0) @ Ueff
#
masses3D = np.c_[masses,masses,masses].reshape(nAtoms * 3) * 1822.888486
SqrtMasses = np.sqrt(masses3D) 
del masses3D
Cm1 = Ueff.T @ np.diag(np.sqrt(w0)) @ U0.T

###############################

def Align(geoStart, CheckAlignment = False):
    """
    Rotate the molecule so to eliminate the projection along
    the infinitesimal translational coordinates
    """
    geo = geoStart.copy()
    # Check the alignment
    if CheckAlignment:
        print('')
        print('ALIGN: Initial components on the rotational coordinates:')
        print(TR[:,(3,4,5)].T @ (SqrtMasses * geo.reshape(nAtoms * 3)))
    #
    maxIter = 200
    iC = 0
    eps = 1e-8
    for i in range(maxIter):
        ixyz = (i % 3) + 3    # Cyclic rotations about the three axes
        if   ixyz == 3:
            geoSwap = np.array([-geo[:,2],  geo[:,1],  geo[:,0]]).T
        elif ixyz == 4:
            geoSwap = np.array([ geo[:,0],  geo[:,2], -geo[:,1]]).T
        elif ixyz == 5:
            geoSwap = np.array([ geo[:,1], -geo[:,0],  geo[:,2]]).T
        #
        b = TR[:,ixyz].T @ (SqrtMasses * geo.reshape(nAtoms * 3))
        c = TR[:,ixyz].T @ (SqrtMasses * geoSwap.reshape(nAtoms * 3))  
        if np.abs(b) < eps:
            iC += 1
        else:
            iC = 0
        if iC == 3: break
        #
        theta = np.arctan(- b / c)
        sRot = np.sin(theta)
        cRot = np.cos(theta)
        geo = cRot * geo + sRot * geoSwap
    if i == maxIter - 1:
        print('WARNING: very distorted geometry.')
        print('It could not be completely aligned.')
    # Check alignment
    if CheckAlignment:
        print('ALIGN: %i iterations' % (i + 1))
        print('ALIGN: Final components on the rotational coordinates:')
        print(TR[:,(3,4,5)].T @ (SqrtMasses * geo.reshape(nAtoms * 3)))        
    #
    return geo.reshape(nAtoms * 3)

def GetQCoords(geo, CheckAlignment = False):
    """
    Converts the geometry geo into dimensionless effective modes
    """
    # Check the shape
    if geo.shape != (24,3):
        print('ERROR: the input geometry should be a (24,3) matrix')
        return None
    else:
        # Center the center of mass to the origin
        RCM = (masses @ geo) / np.sum(masses)
        geoDisp = (geo - RCM)
    #        
    # Align the molecule
    geoDispAligned = Align(geoDisp, CheckAlignment)
    # Evaluate the displacement
    Q = Cm1 @ (SqrtMasses * (geoDispAligned - geoS0))
    #
    return Q