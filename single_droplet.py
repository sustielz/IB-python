#### Script for IBM simulation of a single droplet in incompressible, periodic fluid

### General Setup
import numpy as np 
import warnings; warnings.simplefilter('ignore')
import sys 
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
for key in special_params.keys(): special_params[key] = eval(special_params[key])
locals().update(params)
params.update(special_params)
with open('params.json', 'w') as f:
    json.dump(params, f)
############################

#### Define a pIBM droplet using geometry from util
def pibDROPLET(fluid, RAD, POS, Nb=400, h=None, K=40, Kp=2500, M=None):
    h = h or fluid.h
    X_in = FULL_CIRCLE(h, RAD-h/2., POS)
    X_out = CIRCLE(Nb, RAD, POS)
    drop_in= PIB2(X_in, fluid.N, fluid.h, fluid.dt)
    drop_in.Kp = Kp    
    drop_in.M = M or drop_in.M
    drop_out = IB2(X_out, fluid.N, fluid.h, fluid.dt)
    drop_out.K = K
    return [drop_in, drop_out]


####################################
  ########   Simulation   ########
####################################
fluid = FLUID(N=N, L=L, mu=mu)
fluid.dt = dt
droplets = [pibDROPLET(fluid, rad, pos, h=fluid.h/2, K=K, Kp=Kp, Nb=Nb) for pos in positions]

insides = [drop[0] for drop in droplets]
outsides = [drop[1] for drop in droplets]

solids = []
trash = [solids.extend(drop) for drop in droplets]
for inside in insides:
#     inside.bForce = lambda solid, Y: GRAV(solid, Y) - 1*solid.V + 100*TRAPPING_PLANE(Y, fluid.L)
    
#     inside.bForce = lambda solid, Y:  50*TRAPPING_PLANE(Y, fluid.L)*(1+np.sin(2*np.pi*fluid.t/(solid.dt*100)))
    inside.bForce = lambda solid, Y:  GRAV(solid, Y, theta=theta) + Tamp*TRAPPING_PLANE(Y, fluid.L)*(1+np.sin(2*np.pi*fluid.t/(solid.dt*Tper)))
    


delta = [[] for inside in insides]    ## Keep track of |X-Y|/h
V = [[] for inside in insides]
THETA = [[] for outside in outsides]


U = []
Xout = [[] for outside in outsides]
Xin = [[] for inside in insides]
Y = [[] for inside in insides]
for i in range(Nsteps+1):
    iterate(fluid, solids)
    #### Keeping track of 'interior' properties
    for j, iin in enumerate(insides):
        delta[j].append(np.max(np.linalg.norm(iin.Y - iin.X, axis=1)))
        V[j].append(np.mean(iin.V, axis=0))
    if i%10==0:
        print(i)
        U.append(fluid.u.copy())
        for j, iin in enumerate(insides):
            Xin[j].append(iin.X.copy())
            Y[j].append(iin.Y.copy())
        for j, out in enumerate(outsides):
            Xout[j].append(out.X.copy())

#with open(PATH+'/data.npz', 'wb') as f:
with open('data.npz', 'wb') as f:
    np.savez(f, U=np.array(U), Xin=np.array(Xin), Xout=np.array(Xout), Y=np.array(Y), delta=np.array(delta), V=np.array(V))                
    

