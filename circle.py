import numpy as np 
import matplotlib.pyplot as plt

from ib2 import IB2
#### Generic class for an immersed boundary in a 3D fluid
class CIRCLE(IB2):
   
    def __init__(self, Nb, N, h, K=1., R=None, R0=None):
        print(R)
        print(R0)
        self.R = R or N*h/4.
        self.R0 = R0 or (N*h/2., N*h/2.)
        super(CIRCLE, self).__init__(Nb, N, h, K=K)
    
    def initialize(self):  ## Initialize boundary and velocity
        theta = self.dtheta*np.arange(self.Nb)
        self.X = np.array(self.R0)[:, np.newaxis] + self.R*np.array([np.cos(theta), np.sin(theta)])
        self.X = np.array(self.X, dtype=np.float64)
  
    