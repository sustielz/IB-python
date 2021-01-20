import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as tri

#### Generic class for an immersed boundary in a 2D fluid
class IB3(object):

    def __init__(self, X, N, h, dt, K=1., v=None):
        self.X = X     # Positions of boundary
        self.N = N     # Fluid domain properties                
        self.h = h
        self.dt = dt
        self.K = K      # Elastic stiffness
    
        self.Nb = np.shape(self.X)[0]  # number of boundary points
        self.dtheta = 2*np.pi/np.sqrt(self.Nb)
#         self.dtheta = 2*np.pi/self.Nb        # spacing of boundary points
#         self.kp = (np.arange(self.Nb)+1)%self.Nb       # IB index shifted left
#         self.km = np.arange(self.Nb)-1       # IB index shifted right
    
    def step_XX(self, u): self.XX=self.X+0.5*self.dt*self.interp(u,self.X) # Euler step to midpoint
       
    def step_X(self, uu): self.X+=self.dt*self.interp(uu,self.XX) # full step using midpoint velocity            
    
    @property
    def ff(self): return self.vec_spread(self.Forcesurf(self.XX, self.v) ,self.XX) # Force at midpoint
    
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
        W = np.zeros([N, N, N])

        U=np.zeros([Nb,3])
        s=X/float(h)
        i=np.array(np.floor(s), dtype=int)
        r=s-i
        for k in range(Nb):
#             w = self.phi(r[2, k]).outer(self.phi(r[1, k]).outer(self.phi(r[0, k])))
#             w = np.outer(self.phi(r[2, k]), np.outer(self.phi(r[1, k]), self.phi(r[0, k])))
            w = self.phi(r[k, 0])[:, None, None]*self.phi(r[k, 1])[None, :, None]*self.phi(r[k, 2])[None, None, :]
#             w = self.phi(r[2, k])[None, None, :]*self.phi(r[1, k])[None, :, None]*self.phi(r[0, k])[:, None, None]
#             w = np.einsum('i,j,k',self.phi(r[2, k]), self.phi(r[1, k]), self.phi(r[0, k]))
#             if k==0:
#                 print('w 0:')
#                 print(w)
            i1 = np.arange(i[k,0]-1, i[k,0]+3)%N
            i2 = np.arange(i[k,1]-1, i[k,1]+3)%N
            i3 = np.arange(i[k,2]-1, i[k,2]+3)%N
            iii = np.meshgrid(i1, i2, i3, indexing='ij')
#             if k==0:
#                 print('ii_interp')
#                 print(ii)

            U[k,0]=np.sum(w*u[0][iii]);
            U[k,1]=np.sum(w*u[1][iii]);
            U[k,2]=np.sum(w*u[2][iii]);
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
        W = np.zeros([N, N, N])
        
        c=self.dtheta/h**3;
#         c=1./h**3;
        f=np.zeros([3,N,N,N]);
        s=X/float(h)
        i=np.array(np.floor(s), dtype=int)
        r=s-i
        for k in range(Nb):
            w = self.phi(r[k, 2])[None, None, :]*self.phi(r[k, 1])[None, :, None]*self.phi(r[k, 0])[:, None, None]
            i1 = np.arange(i[k,0]-1, i[k,0]+3)%N
            i2 = np.arange(i[k,1]-1, i[k,1]+3)%N
            i3 = np.arange(i[k,2]-1, i[k,2]+3)%N
            iii = np.meshgrid(i1, i2, i3, indexing='ij')

            f[0][iii]+=(c*F[k,0])*w #Spread force to fluid
            f[1][iii]+=(c*F[k,1])*w
            f[2][iii]+=(c*F[k,2])*w
                       
        return f 

#   #### Methods for visualization/plotting
#     def show_X(self, X=None, L=None):
#         X = X or self.X
#         L = L or self.N*self.h
#         plt.scatter(X[0],X[1])
#         plt.xlim([0,L])
#         plt.ylim([0,L])    
        
    def show_phi(self, X=None, show_X=True):
        N, Nb, h = self.N, self.Nb, self.h
        W = np.zeros([N, N, N])

        X = X or self.X
        s=X/float(h)
        i=np.array(np.floor(s), dtype=int)
        r=s-i
        for k in range(Nb):
            w = self.phi(r[k, 0])[:, None, None]*self.phi(r[k, 1])[None, :, None]*self.phi(r[k, 2])[None, None, :]

            i1 = np.arange(i[k,0]-1, i[k,0]+3)%N
            i2 = np.arange(i[k,1]-1, i[k,1]+3)%N
            i3 = np.arange(i[k,2]-1, i[k,2]+3)%N
            iii = np.meshgrid(i1, i2, i3, indexing='ij')

            W[iii] += w
        return W
            
        
            
    def grad(self,A,B,C):
        base=np.sqrt(np.sum((B-C)**2));
        n=np.cross(B-A,C-A);
        un=n/np.linalg.norm(n, axis=0);
        dct=np.cross(B-C,un);
        dct=dct/np.linalg.norm(dct, axis=0)
        return dct*base/2
        
    def Forcesurf(self, X, v): #this force calculation uses the surface energy(the area) as its energy function.
        Nb = self.Nb
        K = self.K
        F= np.zeros([Nb,3])
        
        numtri=np.shape(v)[0]
        for ti in range(numtri):
            A=X[v[ti,0]]
            B=X[v[ti,1]]
            C=X[v[ti,2]]
            F[v[ti,0]]+=self.grad(A,B,C);
            F[v[ti,1]]+=self.grad(B,A,C);
            F[v[ti,2]]+=self.grad(C,A,B);
       
        return -F*K;
