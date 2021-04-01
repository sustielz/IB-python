import numpy as np 
import matplotlib.pyplot as plt

from ib2 import IB2


#### A penalty immersed boundary method for assigning mass density to a fluid. 

#### Massive fluid 'markers' consist of pairs of points connected by a stiff spring: 
#### a massless point X which moves at the local fluid velocity and applies force to the fluid; 
#### and a massive point Y which is subject to body forces but does not interact with the fluid. 
#### The force applied to the fluid at each point X is determined by the spring force to its respective mass point Y.


class RPIB2(IB2):
    def __init__(self, X, N, h, dt, K=1., Kp=None):
        super(RPIB2, self).__init__(X, N, h, dt, K=K)
        self.Kp = Kp or K
#         self.dtheta = 1.          
        self.Y = self.X.copy()    #### massive points Y initially coincide with fluid markers X
        self.V = self.Y*0.   
        self.YCM = np.mean(self.Y, axis=0)       
        self.VCM = np.zeros(2)

        C = self.C = self.Y - self.YCM[np.newaxis,:]
        self.E = np.eye(2)        
        self.EE = np.eye(2)
        self.M = 0.01     
        
        self.dtheta = 1./self.Nb

#         self.I0 = sum(np.linalg.norm(C, axis=0)**2)*np.eye(2) - np.inner(C, C)
#         self.I0i = np.linalg.inv(self.I0)
        self.I0 = self.M*sum(np.linalg.norm(C, axis=1)**2)     #### Simplified since we only care about Lz
        self.I0i = 1./self.I0
#         self.L = np.zeros(2)
        self.L = 0.
        
    def step_XX(self, u): 
        super(RPIB2, self).step_XX(u)
        self.YYCM = self.YCM+0.5*self.dt*self.VCM

#         Omega = self.E.dot(self.I0i.dot(self.E.transpose().dot(self.L)))
#         omega = np.linalg.norm(Omega)
#         theta = 0.5*self.dt*omega
#         ax = Omega/omega if omega>0 else Omega
#         print('Omega')
#         print(Omega)
#         print('ax')
#         print(ax)          


#         for i in range(2):
#             self.EE[i] = self.rot(ax, theta, self.E[i])
# #         self.EE = self.E#self.rot(ax, theta, self.E)
    
        #### TODO: Is this - sign correct?
        theta = -0.5*self.dt*self.L/self.I0      #### Simplified sinze L=Lz is not a vector
        for i in range(2):
            self.EE[i] = self.rot(theta, self.E[i])
#         self.EE = self.E#self.rot(ax, theta, self.E)
    


        self.YY = self.YCM[np.newaxis, :] + self.C.dot(self.EE)  #  self.EE.dot(self.C)   
        self.VVCM = self.VCM - 0.5*self.dt/self.M*np.mean((self._bForce(self.YY) - self.FF), axis=0)    #### Factor of dtheta cancels from definition of M

        self.LL = self.L + 0.5*self.dt*self.TT

        return self.FF
        
    def step_X(self, uu):  # full step using midpoint velocity            
        super(RPIB2, self).step_X(uu)
        
        self.YCM += self.dt*self.VVCM
        
#         Omega = self.EE.dot(self.I0i.dot(self.EE.transpose().dot(self.LL)))
#         omega = np.linalg.norm(Omega)
#         theta = self.dt*omega
#         ax = Omega/omega if omega>0 else Omega
#         for i in range(2):
#             self.E[i] = self.rot(ax, theta, self.E[i])
# #         self.E = self.rot(ax, theta, self.E)
        
        #### TODO: Is this - sign correct?
        theta = -self.dt*self.L/self.I0      #### Simplified sinze L=Lz is not a vector    
        for i in range(2):
            self.E[i] = self.rot(theta, self.E[i])
#         self.E = self.rot(ax, theta, self.E)
        
        
        self.Y = self.YCM[np.newaxis, :] +  self.C.dot(self.E)  #self.E.dot(self.C)    
        self.VCM += self.dt/self.M*np.mean((self._bForce(self.YY) - self.FF), axis=0)
#         self.VCM[1] -= self.dt*9.8
        
        self.L += self.dt*self.TT
        
        return self.FF
#       






    def _bForce(self, Y):
        return self.bForce(self, Y)
   
    
    def bForce(rpib2, Y):
#         pass
        out = Y*0.
        out[1] -= rpib2.M*980
        return out
        
#         return -980*np.array([0., 1.])
#         return -9.8*np.array([0., 1.])

        
    def pForce(self, Y, X): return self.Kp*(Y-X)
    
    def pTorque(self, Y, YCM, F):
# #         print(YCM)
# #         print(np.shape(Y))
# #         print(np.shape(F))
# #         print(np.shape( np.cross(Y-YCM[:, np.newaxis], F, axisa=0, axisb=0) ))
# #         return (self.dtheta/self.h**2)*np.sum( np.cross(Y-YCM[:, np.newaxis], F, axisa=0, axisb=0), axis=1)
# #         return (self.dtheta/self.h**2)*np.sum( np.cross(Y-YCM[:, np.newaxis], F, axisa=0, axisb=0))
#         return (self.dtheta/self.h**2)*np.sum( np.cross(Y-YCM[:, np.newaxis], F, axisa=0, axisb=0))
        
    
        C = Y-YCM[np.newaxis, :]
#         return (self.dtheta/self.h**2)*np.sum(C[0]*F[1]-C[1]*F[0])
        return np.mean(C[:, 0]*F[:, 1]-C[:, 1]*F[:, 0])
    @property
    def FF(self): return self.pForce(self.YY, self.XX) #+ self.Force(self.XX)
    
    @property
    def ff(self): return self.vec_spread(self.FF, self.XX) # Force at midpoint
    
    @property
    def TT(self): return self.pTorque(self.YY, self.YYCM, -self.FF)
    
    def rot(self, theta, X): 
        
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#         out = rotation.dot(X)
        out = X.dot(rotation)
#         print(theta)
#         print(np.shape(X))
#         print(np.shape(rotation))
#         print(np.shape(out))
#         print('ybuin')
        return out
#     def rot(self, ax, theta, X): return np.cos(theta)*X + (1-np.cos(theta))*np.inner(ax, X)*ax + np.sin(theta)*np.cross(ax, X)

#     def rot(self, ax, theta, X): 
#         shape = list(np.shape(X))
#         dim = shape[0]
#         shape[0] = 3
#         X0 = np.zeros(shape)
#         X0[:dim] = X
#         ax = np.array([ax[0], ax[1], 1e-20])
#         return (np.cos(theta)*X0 + (1-np.cos(theta))*np.inner(ax, X0)*ax + np.sin(theta)*np.cross(ax, X0))
    
    
    