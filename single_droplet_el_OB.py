#### Script for IBM simulation of a single droplet in incompressible, periodic fluid

### General Setup
import numpy as np 
import warnings; warnings.simplefilter('ignore')
import sys
sys.path.append('src')
import os
import json

from fluid2 import FLUID    #### Generic fluid solver
from ib2 import IB2         #### 2D Immersed Boundary object (droplet membrane)
from pib2 import PIB2       #### 2D penalty IB object (droplet interior)
from util import *          #### General functions (iterate, geometry, force functions, etc)

########    I/O    ########
#### Load Default Parameters into Variables ####
with open('default_params.json', 'r') as f:
    params = json.load(f)
locals().update(params)
#### argv are for custom parameters.
#### IN BASH: run as python3 script.py K 100 dt 0.005 ... [paramname] [paramval]
ARGV = sys.argv

special_params = dict(zip(ARGV[1::2], ARGV[2::2]))
FILENAME='trial_'+'_'.join(ARGV[1:]).replace('.', 'pt')
for key in special_params.keys(): special_params[key] = eval(special_params[key])
locals().update(special_params)
params.update(special_params)
with open('params/{}.json'.format(FILENAME), 'w') as f:
    json.dump(params, f)
############################

#### Define a pIBM droplet using geometry from util
class IB2_AC(IB2):
    def __init__(self, fluid, *args, **kwargs):
        super(IB2_AC, self).__init__(*args, **kwargs)
        self.Force = lambda X: self.Force_spring(X) + Tamp*TRAPPING_PLANE(X, fluid.L)*(1+np.sin(2*np.pi*fluid.t/(fluid.dt*Tper)))

#### Define a pIBM droplet using geometry from util
def pibDROPLET(fluid, RAD, POS, Nb=400, K=40, Ni=200, Kp=2500, M=None):
#     X_in = FULL_CIRCLE(RAD-fluid.h/2, POS, Ni)
    X_in = SUNFLOWER(RAD-fluid.h/2, POS, Ni)
    X_out = CIRCLE(RAD, POS, Nb)
    drop_in= PIB2(X_in, fluid.N, fluid.h, fluid.dt)
    drop_in.Kp = Kp    
    drop_in.M = M or drop_in.M
    drop_out = IB2_AC(fluid, X_out, fluid.N, fluid.h, fluid.dt)
    drop_out.K = K
    return [drop_in, drop_out]

####################################
  ########   Simulation   ########
####################################

#### Initialize Fluid+Droplets
fluid = FLUID(N=N, L=L, mu=mu, dt=dt)
droplets = [pibDROPLET(fluid, rad, positions[i], Nb=Nb, K=K, Ni=Ni, Kp=Kp, M=M) for i in range(len(positions))]
# rad = [0.05, 0.1]
# droplets = [pibDROPLET(fluid, rad[i], positions[i], h=fluid.h/2, K=K, Kp=Kp, Nb=Nb, M=M) for i in range(len(positions))]



insides = [drop[0] for drop in droplets]
outsides = [drop[1] for drop in droplets]
solids = []
for drop in droplets:
    solids.append(drop[0])
    solids.append(drop[1])
    
#### Declare Forces
for inside in insides:
#     inside.bForce = lambda solid, Y: GRAV(solid, Y) - 1*solid.V + 100*TRAPPING_PLANE(Y, fluid.L)
    
#     inside.bForce = lambda solid, Y:  50*TRAPPING_PLANE(Y, fluid.L)*(1+np.sin(2*np.pi*fluid.t/(solid.dt*100)))
    inside.bForce = lambda solid, Y:  GRAV(solid, Y, theta=stheta*np.pi) #+ Tamp*TRAPPING_PLANE(Y, fluid.L)*(1+np.sin(2*np.pi*fluid.t/(solid.dt*Tper)))
    
    
    
#### Values that we're tracking

U = []
Xout = [[] for outside in outsides]
Xin = [[] for inside in insides]
Y = [[] for inside in insides]
V = [[] for inside in insides]
for i in range(nsteps+1):
    iterate(fluid, solids)
    #### Keeping track of 'interior' properties
    for j, iin in enumerate(insides):
#         delta[j].append(np.max(np.linalg.norm(iin.Y - iin.X, axis=1)))
        Xin[j].append(iin.X.copy())
        Y[j].append(iin.Y.copy())
        V[j].append(iin.V.copy())
    for j, out in enumerate(outsides): Xout[j].append(out.X.copy())
    if i%nmod==0: 
        print(i)
        U.append(fluid.u.copy())
#         for j, iin in enumerate(insides):
#             Xin[j].append(iin.X.copy())
#             Y[j].append(iin.Y.copy())
#         for j, out in enumerate(outsides):
#             Xout[j].append(out.X.copy())
#with open(PATH+'/data.npz', 'wb') as f:

with open('data/{}.npz'.format(FILENAME), 'wb') as f:
    np.savez(f, U=np.array(U), Xin=np.array(Xin), Xout=np.array(Xout), Y=np.array(Y), V=np.array(V))                
    

