import numpy as np 
import matplotlib.pyplot as plt

from fluid import FLUID
# from ib2 import IB2

class IBSIM(object):
    
    def __init__(self, fluid, solids=[]):
        self.fluid = fluid
        self.solids = solids
        self.t = 0
    

    def add_solid(self, SOLID, Nb, **kwargs): self.solids.append(SOLID(Nb, self.fluid.N, self.fluid.h, **kwargs))
        
    # Run until time t
    def run(self, t):
        while self.fluid.t < t:         #for clock=1:clockmax
            self.iterate()

    def iterate(self):
        dt = self.fluid.dt
        ## Run simulation
        self.fluid.t += dt
        ff = np.zeros(np.shape(self.fluid.u))
        for solid in self.solids:
            solid.XX=solid.X+(dt/2)*solid.interp(self.fluid.u,solid.X) # Euler step to midpoint
            ff += solid.vec_spread(solid.Force(solid.XX),solid.XX) # Force at midpoint
        u,uu=self.fluid.fluid(self.fluid.u,ff) # Step Fluid Velocity
        for solid in self.solids:
            solid.X+=dt*solid.interp(uu,solid.XX) # full step using midpoint velocity            
        self.fluid.u = u

       
    def show_all(self):
        plt.figure(figsize=(10, 10))
        self.fluid.show_vorticity()
        self.fluid.show_streamlines()
        for solid in self.solids:
            solid.show_X()
