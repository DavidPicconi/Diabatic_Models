#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import HBQ
au2eV = 27.21138

# Scan in the (Q1,Q8) plane

Q1 = np.linspace(-2.5,12,30)
Q9 = np.linspace(-4.0,4.0,17)

DiabPot = np.zeros((len(Q1),len(Q9),4,4))
Q = np.zeros(66)  # 66 normal modes

for i1 in range(len(Q1)):
    Q[0] = Q1[i1]
    for i2 in range(len(Q9)):
        Q[8] = Q9[i2]
        DiabPot[i1,i2,:,:] = HBQ.DiabaticPotential_Q(Q).copy() * au2eV
        
        
levels = np.linspace(2.0, 5.0, 16)     
mg = np.meshgrid(Q1,Q9)        

plt.contourf(mg[0], mg[1], DiabPot[:,:,1,1].T,
             levels = levels)
cs = plt.contour(mg[0], mg[1], DiabPot[:,:,1,1].T,
            levels = levels, colors = 'black', linewidths = 0.5)

plt.xlabel('$Q_1$', fontsize = 15)
plt.ylabel('$Q_9$', fontsize = 15)
plt.clabel(cs, inline = True)

plt.show()


# Scan along the OH bond distance
RO = HBQ.geoS0[0:3]     # O atom
RH = HBQ.geoS0[51:54]   # H atom
nOH = (RH - RO) / np.linalg.norm(RH - RO)  # unit vector

rOH = np.linspace(1.0,5.0,33)

DiabPot = np.zeros((len(rOH),4,4))
geoDisp = HBQ.geoS0.copy()
iC = 0
for x in rOH:
    geoDisp[51:54] = RO + x * nOH
    DiabPot[iC,:,:] = HBQ.DiabaticPotential(geoDisp.reshape(24,3), False) * au2eV
    iC += 1
    
plt.plot(rOH, DiabPot[:,0,0], '-o', color = '#000000', label = '$S_0$')
plt.plot(rOH, DiabPot[:,1,1], '-o', color = '#008000', label = '1$\pi\pi^*$')
plt.plot(rOH, DiabPot[:,2,2], '-o', color = '#0000ff', label = '2$\pi\pi^*$')
plt.plot(rOH, DiabPot[:,3,3], '-o', color = '#ff0000', label = '1$n\pi^*$')
plt.legend(fontsize = 15)         
plt.ylim(0.0,6.0)
plt.xlabel('$r_\mathrm{OH} \ [a_0]$', fontsize = 15)
plt.ylabel('energy [eV]', fontsize = 15)
plt.show()