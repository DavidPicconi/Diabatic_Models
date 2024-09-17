"""
Module to read and process the parameter files.

@author: David Picconi
"""

###############################
def Interpolate(scanFile):
    """
    Read the data from datafile and perform a quadratic interpolation
    """
    import os.path
    datapath = os.path.join(os.path.curdir, 'Surface_Parameters')
    #
    from numpy import vectorize
    if not os.path.isfile(os.path.join(datapath, scanFile)):
        return vectorize(lambda x: 0.0)
    #
    with open(os.path.join(datapath, scanFile), 'r') as FileTmp:
        scan = FileTmp.readlines()
    #
    Q1 = [float(x.split()[0]) for x in scan]        
    V  = [float(x.split()[1]) for x in scan]
    #
    from scipy.interpolate import interp1d
    #
    return interp1d(Q1, V, kind = 'quadratic',
                    fill_value = 'extrapolate')
    
    
###############################
def Load_Parameters():
    """
    Read and interpolate all the parameters
    """
    nAtoms = 24
    nModes = 3 * nAtoms - 6
    
    #####
    # One-dimensional cuts
    V1D = [0] * 4
    for iS in range(4):
        scanFile = 'V' + str(iS) + '.dat'
        V1D[iS] = Interpolate(scanFile)
        
    #####
    # Gradients
    grad = [[0 for iM in range(nModes)] for iS in range(4)]
    for iS in range(4):
        for iM in range(nModes - 1):
            scanFile = 'Grad_' + str(iS) + '_' + str(iM + 1) + '.dat'
            grad[iS][iM + 1] = Interpolate(scanFile)
    
    #####
    # Hessians
    Hess = [[[0 for iM in range(nModes)] for jM in range(nModes)]\
              for iS in range(4)]  
    for iS in range(4):
        for iM in range(nModes - 1):
            for jM in range(iM, nModes - 1):
                scanFile = 'Hess_' + str(iS) \
                           + '_' + str(iM + 1) + '_' + str(jM + 1) + '.dat'
                Hess[iS][iM + 1][jM + 1] = Interpolate(scanFile)
                
    #####
    # Cubic and quartic terms
    c3 = [0 for iS in range(4)]
    c4 = [0 for iS in range(4)]
    for iS in range(4):
        scanFile = 'c3_' + str(iS) + '.dat'
        c3[iS] = Interpolate(scanFile)
        scanFile = 'c4_' + str(iS) + '.dat'
        c4[iS] = Interpolate(scanFile)
        
    #####
    # Diabatic couplings
    W12 = Interpolate('coup12.dat')
    W13 = [Interpolate('coup13_1.dat'),
           Interpolate('coup13_2.dat'),
           Interpolate('coup13_3.dat'),
           Interpolate('coup13_4.dat')]
    W23 = [Interpolate('coup23_1.dat'),
           Interpolate('coup23_2.dat'),
           Interpolate('coup23_3.dat'),
           Interpolate('coup23_4.dat')]
    
    ############################
    return V1D, grad, Hess, c3, c4, W12, W13, W23
    