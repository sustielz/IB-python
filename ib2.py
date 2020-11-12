import numpy as np 
import matplotlib.pyplot as plt

#### Generic class for an immersed boundary in a 2D fluid
class IB2(object):

    def __init__(self, X, N, h, dt, K=1.):
        self.X = X     # Positions of boundary
        self.N = N     # Fluid domain properties                
        self.h = h
        self.dt = dt
        self.K = K      # Elastic stiffness
    
        self.Nb = np.shape(X)[1]             # number of boundary points        
        self.dtheta = 2*np.pi/self.Nb        # spacing of boundary points
        self.kp = np.arange(self.Nb)+1       # IB index shifted left
        self.kp[-1] = 0
        self.km = np.arange(self.Nb)-1       # IB index shifted right
    
    def step_XX(self, u): self.XX=self.X+0.5*self.dt*self.interp(u,self.X) # Euler step to midpoint
       
    def step_X(self, uu): self.X+=self.dt*self.interp(uu,self.XX) # full step using midpoint velocity            
    
    @property
    def ff(self): return self.vec_spread(self.Force(self.XX) ,self.XX) # Force at midpoint
    
    def phi(self, r):     ## Discrete dirac delta function
        w = np.zeros(4)
        q=np.sqrt(1+4*r*(1-r))
        w[3]=(1+2*r-q)/8
        w[2]=(1+2*r+q)/8
        w[1]=(3-2*r+q)/8
        w[0]=(3-2*r-q)/8
        return w

    def interp(self, u, X):      ## Interpolate boundary velocity U from fluid velocity u
        N, Nb, h = self.N, self.Nb, self.h
        W = np.zeros([N, N])

        U=np.zeros([2,Nb])
        s=X/float(h)
        i=np.array(np.floor(s), dtype=int)
        r=s-i
        for k in range(Nb):
            w = np.outer(self.phi(r[1, k]), self.phi(r[0, k]))
            i1 = np.arange(i[0,k]-1, i[0,k]+3)%N
            i2 = np.arange(i[1,k]-1, i[1,k]+3)%N
            ii = np.meshgrid(i1, i2)
#             if k==0:
#                 print('ii_interp')
#                 print(ii)

            U[0,k]=np.sum(w*u[0][ii]);
            U[1,k]=np.sum(w*u[1][ii]);
#             W[ii] += w

#         print('after interp:')
#         plt.imshow(np.transpose(W))
#         plt.colorbar()
#         plt.scatter(X[0]/h, X[1]/h)
#         plt.show()
#         print(np.mean(abs(U)))
        return U
    
    def vec_spread(self, F, X):   ## Spread boundary force F onto fluid domain ff
        N, Nb, h = self.N, self.Nb, self.h
        W = np.zeros([N, N])
        
        c=self.dtheta/h**2;
        f=np.zeros([2,N,N]);
        s=X/float(h)
        i=np.array(np.floor(s), dtype=int)
        r=s-i
        for k in range(Nb):
#             w = np.outer(self.phi(r[0, k]), np.flip(self.phi(r[1, k])))
            w = np.outer(self.phi(r[1, k]), self.phi(r[0, k]))
          
            i1 = np.arange(i[0,k]-1, i[0,k]+3)%N
            i2 = np.arange(i[1,k]-1, i[1,k]+3)%N
            ii = np.meshgrid(i1, i2)
#             if k==0:
#                 print('ii_spread')
#                 print(ii)
            
            f[0][ii]+=(c*F[0,k])*w #Spread force to fluid
            f[1][ii]+=(c*F[1,k])*w
#             f[0] = np.transpose(f[0])
#             f[1] = np.transpose(f[1])
#             W[ii] += w
                       
#         print('after spread:')
#         plt.imshow(np.transpose(W))
#         plt.colorbar()
#         plt.scatter(X[0]/h, X[1]/h)
#         plt.show()
        return f 

    # elastic stretching force
    def Force(self, X):
        kp, km, dtheta, K = self.kp, self.km, self.dtheta, self.K
        return K*(X[:,kp]+X[:,km]-2*X)/(dtheta**2);
    
  #### Methods for visualization/plotting
    def show_X(self, X=None, L=None):
        X = X or self.X
        L = L or self.N*self.h
        plt.scatter(X[0],X[1])
        plt.xlim([0,L])
        plt.ylim([0,L])    
        
    def show_phi(self, X=None, show_X=True):
        N, Nb, h = self.N, self.Nb, self.h
        W = np.zeros([N, N])

        X = X or self.X
        s=X/h
        i=np.array(np.floor(s), dtype=int)
        r=s-i
        for k in range(Nb):
            w = np.outer(self.phi(r[1, k]), self.phi(r[0, k]))
            i1 = np.arange(i[0,k]-1, i[0,k]+3)%N
            i2 = np.arange(i[1,k]-1, i[1,k]+3)%N
            ii = np.meshgrid(i2, i1)
      
            W[ii] += w

        plt.imshow(W, origin='lower')
        plt.colorbar()
        if show_X: self.show_X()