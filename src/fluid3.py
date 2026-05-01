import numpy as np 
import matplotlib.pyplot as plt

#### 3D Fluid solver using fft
class FLUID(object):
  

            
    def __init__(self, N=64, L=1., rho=1., mu=.01, dt=.01):
        self._N = N 
        self._L = float(L) 
        self._h = self._L/self._N
        self._rho = rho  # Fluid density
        self._mu = mu    # viscosity 
        self._dt = dt    # Time step        
        
        self.init_a()   # Matrix for use in fluid solver 
        
        self.u = np.zeros([3,N,N,N]) #Fluid velocity
        self.ip = (np.arange(N)+1)%N   # Grid index shifted left
        self.im = np.arange(N)-1   # Grid index shifted right
        self.t = 0      # Time
     
 
    def boundary(self, u): return u  # Override to impose boundary conditions on the fluid
    
    def step_u(self, ff):
        self.u, uu = self.fluid(self.u, ff)
        return uu
        
    def init_a(self):
        N = self.N
        a = np.zeros([3,3,N,N,N])
        a[0, 0] = 1
        a[1, 1] = 1
        a[2, 2] = 1

        for m1 in range(N):
            for m2 in range(N):
                for m3 in range(N):
                    if not ( (m1==0 or (N%2==0 and m1==int(N/2))) 
                            and (m2==0 or (N%2==0 and m2==int(N/2)))
                            and (m3==0 or (N%2==0 and m3==int(N/2)))):
                        t=(2*np.pi/N)*np.array([m1, m2, m3]);
                        s=np.sin(t);

                        #### Note matrix multiplication might matter here
                        ss=np.outer(s, s)/np.inner(s, s)

                        #     a(m1+1,m2+1,:,:)=a(m1+1,m2+1,:,:)-(s*s')/(s'*s)
                        a[0,0,m1,m2,m3]-=ss[0,0]
                        a[0,1,m1,m2,m3]-=ss[0,1]
                        a[0,2,m1,m2,m3]-=ss[0,2]
                        
                        a[1,0,m1,m2,m3]-=ss[1,0]
                        a[1,1,m1,m2,m3]-=ss[1,1]
                        a[1,2,m1,m2,m3]-=ss[1,2]
                        
                        a[2,0,m1,m2,m3]-=ss[2,0]
                        a[2,1,m1,m2,m3]-=ss[2,1]
                        a[2,2,m1,m2,m3]-=ss[2,2]

        for m1 in range(N):
            for m2 in range(N):
                for m3 in range(N):
                    t=(np.pi/N)*np.array([m1, m2, m3]);
                    s=np.sin(t);
                    a[:,:,m1,m2,m3] /= (1+(self.dt/2)*(self.mu/self.rho)*(4/(self.h**2))*(np.inner(s, s)));
        self.a = a
    
    # Second order centered Laplacian
    def laplacian(self, u): 
        im, ip, h = self.ip, self.im, self.h
        w=(u[:,ip,:,:]+u[:,im,:,:]+u[:,:,ip,:]+u[:,:,im,:]+u[:,:,:,ip]+u[:,:,:,im]-6*u)/(h**2);
        return w
    
    
    
    def skew(self, u):
        w=u*1. #note that this is done only to make w the same size as u
        w[0]=self.sk(u,u[0]);
        w[1]=self.sk(u,u[1]);
        w[2]=self.sk(u,u[2]);
        return w
    
    def sk(self, u, g):
        ip, im, h = self.ip, self.im, self.h
        ii = np.arange(self.N)
        return ((u[0][ip,:,:]+u[0])*g[ip,:,:]
                -(u[0][im,:,:]+u[0])*g[im,:,:]
                +(u[1][:,ip,:]+u[1])*g[:,ip,:]
                -(u[1][:,im,:]+u[1])*g[:,im,:]
                +(u[2][:,:,ip]+u[1])*g[:,:,ip]
                -(u[2][:,:,im]+u[1])*g[:,:,im])/(6*h);

    
    # Time step the fluid
    def fluid(self, u, ff): 
        self.t += self.dt
        self.boundary(u)
        uu = np.zeros(np.shape(u), dtype=np.complex)
        uuu = np.zeros(np.shape(u), dtype=np.complex)
        
        a, dt, rho, mu = self.a, self.dt, self.rho, self.mu
        w=u-(dt/2)*self.skew(u)+(dt/(2*rho))*ff; # Get RHS
        w=np.fft.fft(w, axis=1)
        w=np.fft.fft(w, axis=2)
        w=np.fft.fft(w, axis=3)
        uu[0]=a[0,0]*w[0]+a[0,1]*w[1]+a[0,2]*w[2] # Solve for LHS
        uu[1]=a[1,0]*w[0]+a[1,1]*w[1]+a[1,2]*w[2] 
        uu[2]=a[2,0]*w[0]+a[2,1]*w[1]+a[2,2]*w[2] 
        uu=np.fft.ifft(uu,axis=3)
        uu=np.fft.ifft(uu,axis=2)
        uu=np.fft.ifft(uu,axis=1).real #Get u at midpoint in time
        self.boundary(uu)
        
        w=u-dt*self.skew(uu)+(dt/rho)*ff+(dt/2)*(mu/rho)*self.laplacian(u)# Get RHS
        w=np.fft.fft(w, axis=1)
        w=np.fft.fft(w, axis=2)
        w=np.fft.fft(w, axis=3)
        uuu[0]=a[0,0]*w[0]+a[0,1]*w[1]+a[0,2]*w[2]# Solve for LHS
        uuu[1]=a[1,0]*w[0]+a[1,1]*w[1]+a[1,2]*w[2]
        uuu[2]=a[2,0]*w[0]+a[2,1]*w[1]+a[2,2]*w[2]
        
#         ppp=b[0]*w[0]+b[1]*w[1]+b[2]*w[2]
#         ppp=np.fft.ifft(ppp,axis=3)
#         ppp=np.fft.ifft(ppp,axis=2)
#         ppp=np.fft.ifft(ppp,axis=1).real # Get pressure output
        
        uuu=np.fft.ifft(uuu,axis=3)
        uuu=np.fft.ifft(uuu,axis=2)
        uuu=np.fft.ifft(uuu,axis=1).real # Get u at next timestep
        return uuu, uu
   
  
    
    
  #### Fluid domain properties  ####    
    @property  # Grid Points
    def N(self): return self._N
    
    
    @property  # Grid spacing
    def h(self): return self._h
    
    @property  # Box size
    def L(self): return self._L
    
    @L.setter
    def L(self, L):
        self._L = float(L)
        self._h = self.L/self.N
        self.init_a()
    
    @property
    def rho(self): return self._rho
    
    @rho.setter
    def rho(self, rho):
        self._rho = rho
        self.init_a()
        
    @property
    def mu(self): return self._mu
    
    @mu.setter
    def mu(self, mu):
        self._mu = mu
        self.init_a()

    @property
    def dt(self): return self._dt
    
    @dt.setter
    def dt(self, dt):
        self._dt = dt
        self.init_a()
        
        

        
        
