import numpy as np 
import matplotlib.pyplot as plt

#### Generic class for an immersed boundary in a 3D fluid
class IB2(object):
  
  #### Fluid properties ####
    @property  # Number of grid cells
    def N(self): return self._N
    
    @N.setter
    def N(self, N): self._N = int(N)
   
    @property  # Grid spacing
    def h(self): return self._h
      
    @h.setter
    def h(self, h): self._h = float(h)
        
  #### Immersed Boundary domain roperties ####    
    @property  # Number of IB points
    def Nb(self): return self._Nb
    
    @Nb.setter 
    def Nb(self, Nb):
        self._Nb = int(Nb)
        self._dtheta = 2*np.pi/Nb        
        self.kp = np.arange(self.Nb)+1       # IB index shifted left
        self.kp[-1] = 0
        self.km = np.arange(self.Nb)-1       # IB index shifted right
    
    @property  # IB point spacing
    def dtheta(self): return self._dtheta
  
    def __init__(self, Nb, N, h, K=1.):
        self.N = N                         
        self.h = h
        self.Nb = Nb  
        self.K = K      # Elastic stiffness
    
        self.initialize()
    
    def initialize(self):  ## Initialize boundary and velocity
        theta = self.dtheta*np.arange(self.Nb)
        self.X = (self.N*self.h/2) + (self.N*self.h/4)*np.array([np.cos(theta), np.sin(theta)])
        self.X = np.array(self.X, dtype=np.float64)

    
    def phi(self, r):
        w = np.zeros(4)
        q=np.sqrt(1+4*r*(1-r))
        w[3]=(1+2*r-q)/8
        w[2]=(1+2*r+q)/8
        w[1]=(3-2*r+q)/8
        w[0]=(3-2*r-q)/8
        return w

    def interp(self, u, X):
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
    
    def vec_spread(self, F, X):
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
            
            f[0][ii]+=(c*F[0,k])*w #Spread force to fluid
            f[1][ii]+=(c*F[1,k])*w
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
        
    def show_phi(self, X=None):
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
            ii = np.meshgrid(i1, i2)
      
            W[ii] += w

        plt.imshow(np.transpose(W))
        plt.colorbar()
        self.show_X()