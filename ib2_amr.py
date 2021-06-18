import numpy as np 
import matplotlib.pyplot as plt

from ib2 import IB2         

class IB2_AMR(IB2):
    
    def __init__(self, X, N, h, dt, **kwargs):
        super(IB2_AMR, self).__init__(X, N, h, dt, **kwargs)
        self.Force = self.Force_surf
        self.a = self.dt/6.
    
    def Force_surf(self, X):
        K, kp, km = self.K, self.kp, self.km
        dX = X - X[km]
        lX = np.linalg.norm(dX, axis=1)
        return self.K*(dX[kp]/lX[kp, np.newaxis] - dX/lX[:, np.newaxis])/(self.dtheta)
             
    def step_X(self, uu): 
        super(IB2_AMR, self).step_X(uu)
        K, kp, km = self.K, self.kp, self.km
        for i in range(10):
            X = self.X
            dX = X - X[km]
            lX = np.linalg.norm(dX, axis=1)

            Fn = dX[kp]/lX[kp, np.newaxis] - dX/lX[:, np.newaxis]
            un = Fn/np.linalg.norm(Fn, axis=1)[:, np.newaxis]
            ut = np.matmul(un, np.array([[0, -1], [1, 0]]))
            uFFt = self.Force_spring(X)/self.K
            self.FFt = np.sum(ut*uFFt, axis=1)[:, np.newaxis]*ut
    #         self.FFt[np.linalg.norm(self.FFt, axis=1) < 1e-1] = 0
            self.X += self.a*self.FFt
   
   
# class IB2(object):

#     @property
#     def Nb(self): return self._Nb
    
#     @Nb.setter
#     def Nb(self, Nb): 
#         self._Nb = Nb
#         self.kp = (np.arange(Nb)+1)%Nb       # IB index shifted left
#         self.km = np.arange(Nb)-1            # IB index shifted right
    
#     @property
#     def dtheta(self): return 2*np.pi/self.Nb
    
    
#     def __init__(self, X, N, h, dt, K=1.):
#         self.X = X     # Positions of boundary
#         self.Nb = np.shape(X)[0]             # number of boundary points       

#         self.N = N     # Fluid domain properties                
#         self.h = h
#         self.dt = dt
#         self.K = K      # Elastic stiffness
#         self.Force = self.Force_surf
    
    
#     def step_XX(self, u): 
#         lX = np.linalg.norm(self.X - self.X[self.km], axis=1)
#         ii = self.km+1
#         self.adj_X(list(ii[lX>0.7*self.h]), list(ii[lX<0.3*self.h]))
        
#         self.XX=self.X+0.5*self.dt*self.interp(u,self.X) # Euler step to midpoint

       
#     def step_X(self, uu): 
#         self.X+=self.dt*self.interp(uu,self.XX) # full step using midpoint velocity  
        
#     @property
#     def ff(self): return self.vec_spread(self.Force(self.XX), self.XX) # Force at midpoint
    
#     def Force_surf(self, X):
#         K, kp, km = self.K, self.kp, self.km
#         dX = X - X[km]
#         lX = np.linalg.norm(dX, axis=1)
# #         self.lXX = lX
#         return self.K*(dX[kp]/lX[kp, np.newaxis] - dX/lX[:, np.newaxis])/(self.dtheta)
    
    
    
#     def phi(self, r):     ## Discrete dirac delta function
#         w = np.zeros(4)
#         q=np.sqrt(1+4*r*(1-r))
#         w[3]=(1+2*r-q)/8
#         w[2]=(1+2*r+q)/8
#         w[1]=(3-2*r+q)/8
#         w[0]=(3-2*r-q)/8
#         return w

#     def interp(self, u, X):      ## Interpolate boundary velocity U from fluid velocity u
#         N, Nb, h = self.N, self.Nb, self.h
#         W = np.zeros([N, N])

#         U=np.zeros([Nb,2])
#         s=X/float(h)
#         i=np.array(np.floor(s), dtype=int)
#         r=s-i
#         for k in range(Nb):
#             w = self.phi(r[k, 0])[None, :]*self.phi(r[k, 1])[:, None]
#             i1 = np.arange(i[k,0]-1, i[k,0]+3)%N
#             i2 = np.arange(i[k,1]-1, i[k,1]+3)%N
#             ii = np.meshgrid(i1, i2)

#             U[k,0]=np.sum(w*u[0][ii]);
#             U[k,1]=np.sum(w*u[1][ii]);
#         return U
    
#     def vec_spread(self, F, X):   ## Spread boundary force F onto fluid domain ff
#         N, Nb, h = self.N, self.Nb, self.h
#         W = np.zeros([N, N])
        
#         c=self.dtheta/h**2;
#         f=np.zeros([2,N,N]);
#         s=X/float(h)
#         i=np.array(np.floor(s), dtype=int)
#         r=s-i
#         for k in range(Nb):
#             w = self.phi(r[k, 0])[None, :]*self.phi(r[k, 1])[:, None]
          
#             i1 = np.arange(i[k,0]-1, i[k,0]+3)%N
#             i2 = np.arange(i[k,1]-1, i[k,1]+3)%N
#             ii = np.meshgrid(i1, i2)

#             f[0][ii]+=(c*F[k,0])*w #Spread force to fluid
#             f[1][ii]+=(c*F[k,1])*w
# #         print(max(f))
#         return f 


#     def split_X(self, indices):
#         indices.sort(reverse=True)
#         X = [x for x in self.X]  ## It's ~10x faster to convert to list and use python insert than to use numpy insert
#         for i in indices:    X.insert(i, 0.5*(X[i] + X[i-1]))
#         self.X = np.vstack(X)
#         self.Nb = self.Nb + len(indices)
        
        
#     def merge_X(self, indices):
#         indices.sort(reverse=True)
#         X = [x for x in self.X]  ## It's ~10x faster to convert to list and use python insert than to use numpy insert
#         for i in indices:    
# #             X[i-1] = 0.5*(X[i] + X[i-1])
#             del X[i]
#         self.X = np.vstack(X)
#         self.Nb = self.Nb - len(indices)
        
#     def adj_X(self, split, merge):
#         self.split_X(split)
# #         split.sort(reverse=True) # unnecessary because split_X sorts indices
#         merge = np.array(merge)
#         for i in split: merge[merge>i] +=1
#         merge = list(merge)
#         self.merge_X(merge)
        
# #     # elastic stretching force
# #     def Force(self, X):
# #         kp, km, dtheta, K = self.kp, self.km, self.dtheta, self.K
# #         return K*(X[kp]+X[km]-2*X)/(dtheta**2)
    
#   #### Methods for visualization/plotting
#     def show_X(self, X=None, L=None):
#         X = X or self.X
#         L = L or self.N*self.h
#         plt.scatter(X[:,0],X[:,1])
#         plt.xlim([0,L])
#         plt.ylim([0,L])    
        
#     def show_phi(self, X=None, show_X=True):
#         N, Nb, h = self.N, self.Nb, self.h
#         W = np.zeros([N, N])

#         X = X or self.X
#         s=X/h
#         i=np.array(np.floor(s), dtype=int)
#         r=s-i
#         for k in range(Nb):
#             w = self.phi(r[k, 0])[None, :]*self.phi(r[k, 1])[:, None]
# #             w = np.outer(self.phi(r[1, k]), self.phi(r[0, k]))
#             i1 = np.arange(i[k,0]-1, i[k,0]+3)%N
#             i2 = np.arange(i[k,1]-1, i[k,1]+3)%N
#             ii = np.meshgrid(i2, i1)
      
#             W[ii] += w

#         plt.imshow(W, origin='lower')
#         plt.colorbar()
#         if show_X: 
#             self.show_X()
