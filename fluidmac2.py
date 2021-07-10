import numpy as np 
import matplotlib.pyplot as plt

#### 2D fluid solver on a staggered grid. Algorithms taken from DFIB branch of the stochasticHydroTools github repo.


def phi(r):     ## Discrete dirac delta function
    w = np.zeros(4)
    q=np.sqrt(1+4*r*(1-r))
    w[3]=(1+2*r-q)/8
    w[2]=(1+2*r+q)/8
    w[1]=(3-2*r+q)/8
    w[0]=(3-2*r-q)/8
    return w

class FLUIDMAC(object):
  
  #### Fluid domain properties ####    
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

            
    def __init__(self, N=64, L=1., rho=1., mu=.01, dt=.01):
        self._N = N 
        self.L = L 
        self.ip = (np.arange(N)+1)%N   # Grid index shifted left
        self.im = np.arange(N)-1   # Grid index shifted right
                            
        self.u = np.zeros([2,N,N]) #Fluid velocity
        self.rho = rho  # Fluid density
        self.mu = mu    # viscosity 
        
        self.t = 0      # Time
        self.dt = dt    # Time step        
        
        self.init_Lhat()   # Matrix for use in fluid solver
    
    def boundary(self, u): return u  # Override to impose boundary conditions on the fluid
    
    def step_u(self, ff):
        self.u, uu = self.fluid(self.u, ff)
        return uu
    
#     def init_a(self):
#         N = self.N
#         a = np.zeros([2,2,N,N])
#         a[0, 0] = 1
#         a[1, 1] = 1
#         for m1 in range(N):
#             for m2 in range(N):
#                 t=(np.pi/N)*np.array([m1, m2])
#                 if m1*abs(m1-N/2) + m2*abs(m2-N/2) > 1e-6: 
#                     s=np.sin(2*t)
#                     ss=np.outer(s, s)/np.inner(s, s)
#                     a[:,:,m1,m2]-=ss
#                 s = np.sin(t)
#                 a[:,:,m1,m2] /= (1+(self.dt/2)*(self.mu/self.rho)*(4/(self.h**2))*(np.inner(s, s)))
#         self.a = a
        
    def init_Lhat(self):
        N = self.N
        Lhat = np.zeros([N,N])
        for m1 in range(N):
            for m2 in range(N):
#                 if m1*abs(m1-N/2) + m2*abs(m2-N/2) > 1e-6: 
                s=np.sin((np.pi/N)*np.array([m1, m2]))
                Lhat[m1, m2] += np.inner(s,s)
        Lhat *= -4/self.h**2
        self.Lhat1 = Lhat#; self.Lhat1[0,0]=1
        self.Lhat2 = 1*self.Lhat1; self.Lhat2[0,0]=1
    
    
    # Second order centered Laplacian
    def laplacian(self, u): 
        im, ip, h = self.ip, self.im, self.h
        w=(u[:,ip,:]+u[:,im,:]+u[:,:,ip]+u[:,:,im]-4*u)/(h**2)
        return w
    
    # Solve 2D Poisson eq. using Lhat2
    def poisson2(self, q):
        q=1*q
        q = np.fft.fft(q, axis=0)
        q = np.fft.fft(q, axis=1)
        q = q/self.Lhat2
        q=np.fft.ifft(q,axis=1)
        return np.fft.ifft(q,axis=0).real #Get u at midpoint in time
        
    def interpuu(self, u):
#         N = np.shape(u)[-1]
        N = self.N
        II = [(np.arange(N)+j)%N for j in range(-1, 3)]
        w = phi(0.5)[:, None]*phi(0.5)[None, :]

        uux, uuy = u*0
        for j, ix in enumerate(II):
            for k, iy in enumerate(II):
                iix = np.meshgrid(ix, iy-1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                iiy = np.meshgrid(ix-1, iy)
                uux += u[0][iix].transpose()*w[j,k]
                uuy += u[1][iiy].transpose()*w[k,j]
    #     return np.vstack([ux, uy])
        return uux, uuy
    
######   Poisseuille Benchmark shows that this skew is giving problems
#     def skew(self, u):
#         uux, uuy = self.interpuu(u) ## Interpolate velocity on the staggered grid; uu = u @ (i+1/2)h, (j+1/2)h
#         w=u*1.               
#         w[0]=self.sk([u[0], uux], u[0]) ## On grid x, u_x is given and u_y is interpolated
#         w[1]=self.sk([uuy, u[1]], u[1])
#         return w
    
#     def sk(self, u, g):
#         ip, im, h = self.ip, self.im, self.h
#         ii = np.arange(self.N)
#         return ((u[0][ip,:]+u[0])*g[ip,:]
#                 -(u[0][im,:]+u[0])*g[im,:]
#                 +(u[1][:,ip]+u[1])*g[:,ip]
#                 -(u[1][:,im]+u[1])*g[:,im])/(4*h)

######## from DFIB repo off stochasticHydroTools
##     def advection2D(self, u):
    def skew(self, u):
        N, h, ip, im = self.N, self.h, self.ip, self.im
        
        adv = 1.*u
        ## 1st component of advection
        Dux = np.array([u[0][ip,:]-u[0][im,:], u[0][:, ip]-u[0][:, im]])/(2*h)
        u2i = (u[1][im,:]+u[1])/2
        u2i = (u2i+u2i[:,ip])/2
        u1sq=u[0]**2; Dxu1sq=(u1sq[ip,:]-u1sq[im,:])/(2*h)
        u1u2=u[0]*u2i; Dyu1u2=(u1u2[:,ip]-u1u2[:,im])/(2*h)
        adv[0]=0.5*(u[0]*Dux[0]+u2i*Dux[1])+0.5*(Dxu1sq+Dyu1u2)

        ## 2nd component of advection
        u1i = (u[0][:,im]+u[0])/2
        u1i = (u1i[ip,:]+u1i)/2
        Dxu2= (u[1][ip,:]-u[1][im,:])/(2*h)
        Dyu2= (u[1][:,ip]-u[1][:,im])/(2*h)
        u1u2= u1i*u[1]; Dxu1u2=(u1u2[ip,:]-u1u2[im,:])/(2*h)
        u2sq= u[1]**2; Dyu2sq=(u2sq[:,ip]-u2sq[:,im])/(2*h)
        adv[1]=0.5*(u1i*Dxu2 + u[1]*Dyu2) + 0.5*(Dxu1u2 + Dyu2sq)
        return adv
    
    
#     def skew(self, u):
#         w=u*1. #note that this is done only to make w the same size as u
#         w[0]=self.sk(u,u[0])
#         w[1]=self.sk(u,u[1])
#         return w
    
#     def sk(self, u, g):
#         ip, im, h = self.ip, self.im, self.h
#         ii = np.arange(self.N)
#         return ((u[0][ip,:]+u[0])*g[ip,:]
#                 -(u[0][im,:]+u[0])*g[im,:]
#                 +(u[1][:,ip]+u[1])*g[:,ip]
#                 -(u[1][:,im]+u[1])*g[:,im])/(4*h)

#     def D_h(self, q): return np.array([ q[0][self.ip, :]-q[0], q[1][:,self.ip]-q[1] ])/self.h
   
    # Time step the fluid
    def fluid(self, u, ff):
        self.t += self.dt
        self.boundary(u)
        uu = np.zeros(np.shape(u), dtype=np.complex)
        uuu = np.zeros(np.shape(u), dtype=np.complex)
        Lhat1, dt, rho, mu, ip, im, h = self.Lhat1, self.dt, self.rho, self.mu, self.ip, self.im, self.h
        
        #### Solve a poisson equation for q
        w = u - (dt/2)*self.skew(u) + (dt/(2*rho))*ff # Get RHS
        divw = (w[0][self.ip, :]-w[0])/h + (w[1][:,self.ip]-w[1])/h
        q = self.poisson2(divw)
        self.pp1 = q*2*rho/dt
        #### Solve for each uu
        r = w - np.array([q-q[im,:], q-q[:, im]])/h
        r = np.fft.fft(r, axis=1)
        r = np.fft.fft(r, axis=2)
        uu = r/(1 - 0.5*mu*dt*Lhat1/rho)  
        uu = np.fft.ifft(uu,axis=2)
        uu = np.fft.ifft(uu,axis=1).real #Get u at midpoint in time
        self.boundary(uu)
        
        #### Solve a poisson equation for q
        w = u - dt*self.skew(uu) + (dt/rho)*ff + (dt/2)*(mu/rho)*self.laplacian(u) # Get RHS
        divw = (w[0][self.ip, :]-w[0])/h + (w[1][:,self.ip]-w[1])/h
        q = self.poisson2(divw)
        self.pp2 = q*rho/dt
        #### Solve for each uuu
        r = w - np.array([q-q[im,:], q-q[:, im]])/h
        r = np.fft.fft(r, axis=1)
        r = np.fft.fft(r, axis=2)
        uuu = r/(1 - 0.5*mu*dt*Lhat1/rho)  
        uuu = np.fft.ifft(uuu,axis=2)
        uuu = np.fft.ifft(uuu,axis=1).real # Get u at next timestep
        return uuu, uu
   
    @property
    def vorticity(self):
        u, h, ip, im = self.u, self.h, self.ip, self.im
        ii = im+1
#         return (u[1,ip,:]-u[1,im,:]-u[0,:,ip]+u[0,:,im])/(2*h)
        vorticity=(u[1][np.meshgrid(ip, ii)]
                       -u[1][np.meshgrid(im,ii)]
                       -u[0][np.meshgrid(ii,ip)]
                       +u[0][np.meshgrid(ii,im)])/(2*self.h)
        return vorticity


        
        
