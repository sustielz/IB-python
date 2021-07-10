import numpy as np 
import matplotlib.pyplot as plt

from ib2 import IB2         

#### Class for an immersed boundary in a fluid. The fluid solver and interpolation/spreading use a staggered-grid (MAC) scheme
#### to improve volume conservation.
    
class IBMAC2(IB2):

    def interp(self, u, X): ## Interpolate boundary velocity U from fluid velocity u, with shifts to account for staggered grid
        N, Nb, h = self.N, self.Nb, self.h
        U=np.zeros([Nb, 2])
        shift = np.eye(2)*h/2
        for i, ui in enumerate(u):  U[:,i] = self.interpi(ui, X+shift[i][np.newaxis, :])
        return U
    
    def interpi(self, u, X):  ## Interpolate a single component
        N, Nb, h = self.N, self.Nb, self.h
        U=np.zeros([Nb])
        s=X/float(h)
        i=np.array(np.floor(s), dtype=int)
        r=s-i
        for k in range(Nb):
            w = self.phi(r[k, 0])[None, :]*self.phi(r[k, 1])[:, None]
            i1 = np.arange(i[k,0]-1, i[k,0]+3)%N
            i2 = np.arange(i[k,1]-1, i[k,1]+3)%N
            ii = np.meshgrid(i1, i2)
            U[k]=np.sum(w*u[ii]);
        return U    
   

    def vec_spread(self, F, X):  ## Spread boundary force F onto fluid domain ff
        N, Nb, h = self.N, self.Nb, self.h
        ff=np.zeros([2,N,N]);
        shift = np.eye(2)*h/2
        for i in range(2): ff[i] = self.spreadi(F[:,i], X+shift[i][np.newaxis, :])
        return ff
    
    def spreadi(self, F, X):  ## Spread a single component
        N, Nb, h = self.N, self.Nb, self.h
        c=self.dtheta/h**2;
        ff=np.zeros([N,N]);
        s=X/float(h)
        i=np.array(np.floor(s), dtype=int)
        r=s-i
        for k in range(Nb):
            w = self.phi(r[k, 0])[None, :]*self.phi(r[k, 1])[:, None]  
            i1 = np.arange(i[k,0]-1, i[k,0]+3)%N
            i2 = np.arange(i[k,1]-1, i[k,1]+3)%N
            ii = np.meshgrid(i1, i2)
            ff[ii]+=(c*F[k])*w #Spread force to fluid
        return ff