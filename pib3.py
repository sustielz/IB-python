import numpy as np 
import matplotlib.pyplot as plt

from ib3 import IB3


#### A penalty immersed boundary method for assigning mass density to a fluid. 

#### Massive fluid 'markers' consist of pairs of points connected by a stiff spring: 
#### a massless point X which moves at the local fluid velocity and applies force to the fluid; 
#### and a massive point Y which is subject to body forces but does not interact with the fluid. 
#### The force applied to the fluid at each point X is determined by the spring force to its respective mass point Y.


class PIB3(IB3):
    
    @property
    def dtheta(self): return self.M/self.Nb
    
    def __init__(self, X, N, h, dt, K=1., M=0.01, Kp=None):
        super(PIB3, self).__init__(X, N, h, dt, K=K)
        self.Kp = Kp or K
        self.Y = self.X.copy()    #### massive points Y initially coincide with fluid markers X
        self.V = self.Y*0.
        self.M = M     

        
    def step_XX(self, u): 
        super(PIB3, self).step_XX(u)
        self.YY = self.Y + 0.5*self.dt*self.V
        self.VV = self.V + (0.5*self.dt/self.M)*(self._bForce(self.YY) - self.FF)
        return self.FF
        
    def step_X(self, uu):  # full step using midpoint velocity            
        super(PIB3, self).step_X(uu)
        self.Y += self.dt*self.VV
        self.V += (self.dt/self.M)*(self._bForce(self.YY) - self.FF)
        return self.FF
#       
    def _bForce(self, Y):
        return self.bForce(self, Y)
   
    
    def bForce(pib3, Y):
#         pass
        out = Y*0.
        out[1] -= pib3.M*980
        return out
        
#         return -980*np.array([0., 1.])
#         return -9.8*np.array([0., 1.])
            
    def pForce(self, Y, X): return self.Kp*(Y-X)
    
    #### Penalty force to be spread to fluid
    @property
    def FF(self): return self.pForce(self.YY, self.XX) #+ self.Force(self.XX)
    
    @property
    def ff(self): return self.vec_spread(self.FF, self.XX) # Force at midpoint
    
    
   
    
    
    
