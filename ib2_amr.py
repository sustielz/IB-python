import numpy as np 
import matplotlib.pyplot as plt

from ib2 import IB2         

#### Class for a 2D immersed boundary with surface tension in fluid. In addition to replacing the force functional, 
#### surface tension also requires even spacing of boundary points. This is accomplished here using 'virtual springs', which
#### do not exert force on the fluid but guide tangential motion along the boundary to adjust spacing. 

class IB2_AMR(IB2):
    
    def __init__(self, X, N, h, dt, **kwargs):
        super(IB2_AMR, self).__init__(X, N, h, dt, **kwargs)
        self.Force = self.Force_surf
        self.a = self.dt/6.         ## stiffness of vitrual springs. Note there is a stability restraint a ~< dt/4
        self.n_max = 50000
        self.n_tol = 0.1

    def Force_surf(self, X):  ## Surface tension
        K, kp, km = self.K, self.kp, self.km
        dX = X - X[km]
        lX = np.linalg.norm(dX, axis=1)
        return self.K*(dX[kp]/lX[kp, np.newaxis] - dX/lX[:, np.newaxis])/(self.dtheta)
    
             
    def step_X(self, uu): 
        super(IB2_AMR, self).step_X(uu)
        K, kp, km = self.K, self.kp, self.km
        n_ref = 0
        n_tol = self.n_tol
        while n_tol > self.n_tol*self.h and n_ref < self.n_max:   ## Adjust tangential position until tolerance is reached
            X = self.X
            dX = X - X[km]
            lX = np.linalg.norm(dX, axis=1)
            n_tol = max(lX) - min(lX)
            n_ref += 1

            Fn = dX[kp]/lX[kp, np.newaxis] - dX/lX[:, np.newaxis]
            un = Fn/np.linalg.norm(Fn, axis=1)[:, np.newaxis]
            ut = np.matmul(un, np.array([[0, -1], [1, 0]]))
            uFFt = self.Force_spring(X)/self.K
            self.FFt = np.sum(ut*uFFt, axis=1)[:, np.newaxis]*ut
    #         self.FFt[np.linalg.norm(self.FFt, axis=1) < 1e-1] = 0
            self.X += self.a*self.FFt
#         print(n_ref)       #### DEBUG ####

   
