import numpy as np 
import matplotlib.pyplot as plt

#### Fluid solver for periodic boundary conditions using fft
class FLUID(object):
  
  #### Fluid domain properties ####
    @property  # Number of grid cells
    def N(self): return self._N
    
    @N.setter
    def N(self, N):
        self._N = int(N)
        self._h = self.L/(self.N)
        self.ip = np.arange(N)+1   # Grid index shifted left
        self.ip[-1] = 0
        self.im = np.arange(N)-1   # Grid index shifted right
    
    @property  # Box size
    def L(self): return self._L
    
    @L.setter
    def L(self, L):
        self._L = float(L)
        self._h = self.L/self.N

    @property  # Grid spacing
    def h(self): return self._h
            
    def __init__(self, N=64, L=1., rho=1., mu=.01, dt=.01):
        self._L = L 
        print(self.L)
        self.N = N                         
        self.rho = rho  # Fluid density
        self.mu = mu    # viscosity 
        
        self.dt = dt                    # Time step
    #     tmax=1 # Run until time
    #     clockmax=np.ceil(tmax/dt)
        
        
        self.init_a()
        self.initialize()
    
    def initialize(self):  ## Initialize boundary and velocity
        self.t = 0.        ## Current time
        N, L, ip, im, = self.N, self.L, self.ip, self.im
        u = np.zeros([2,N,N]);
        j = np.arange(N)
        u[1]+=np.sin(2*np.pi*self.h*j/L)[:, np.newaxis]
#         u[1]+=np.sin(2*np.pi*self.h*j/L)[:, np.newaxis]
#         u[0]+=np.cos(2*np.pi*self.h*j/L)
        u[1] = np.transpose(u[1])
        u[0] = np.transpose(u[0])
        self.u = u

        # %% Initialize animation
        vorticity=(u[1,ip,:]-u[1,im,:]-u[0,:,ip]+u[0,:,im])/(2*self.h);
        dvorticity=(np.max(vorticity)-np.min(vorticity))/5;
        dvorticity = max(dvorticity, 0.1)  ## Catch error on 0 (or uniform) vorticity
        # values= (-10*dvorticity):dvorticity:(10*dvorticity); % Get vorticity contours
        values= np.arange(-10*dvorticity, 10*dvorticity, dvorticity); # Get vorticity contours
        valminmax=[min(values),max(values)];
            
    def init_a(self):
        N = self.N
        a = np.zeros([2,2,N,N])#, dtype=np.complex)#, dtype=np.complex128)
        a[0, 0] = 1
        a[1, 1] = 1

        for m1 in range(N):
            for m2 in range(N):
                if not ((m1==0 or (N%2==0 and m1==int(N/2))) and (m2==0 or m2==(N%2==0 and int(N/2)))):
                    t=(2*np.pi/N)*np.array([m1, m2]);
                    s=np.sin(t);

                    #### Note matrix multiplication might matter here
                    ss=np.outer(s, s)/np.inner(s, s)

                    #     a(m1+1,m2+1,:,:)=a(m1+1,m2+1,:,:)-(s*s')/(s'*s)
                    a[0,0,m1,m2]-=ss[0,0]
                    a[0,1,m1,m2]-=ss[0,1]
                    a[1,0,m1,m2]-=ss[1,0]
                    a[1,1,m1,m2]-=ss[1,1]

        for m1 in range(N):
            for m2 in range(N):
                t=(np.pi/N)*np.array([m1, m2]);
                s=np.sin(t);
                a[:,:,m1,m2] /= (1+(self.dt/2)*(self.mu/self.rho)*(4/(self.h**2))*(np.inner(s, s)));
        self.a = a
    
    # Second order centered Laplacian
    def laplacian(self, u): 
        im, ip, h = self.ip, self.im, self.h
        w=(u[:,ip,:]+u[:,im,:]+u[:,:,ip]+u[:,:,im]-4*u)/(h**2);
        return w
    #     % im = [N,1:(N-1)] = circular version of i-1
    #     % ip = [2:N,1]     = circular version of i+1
    #     % N  = number of points in each space direction
    
    
    def skew(self, u):
        w=u*.1; #note that this is done only to make w the same size as u
        w[0]=self.sk(u,u[0]);
        w[1]=self.sk(u,u[1]);
        return w
    
    
    def sk(self, u, g):
        ip, im, h = self.ip, self.im, self.h
        return ((u[0,ip,:]+u[0])*g[ip,:]
                -(u[0,im,:]+u[0])*g[im,:]
                +(u[1,:,ip]+u[1])*g[:,ip]
                -(u[1,:,im]+u[1])*g[:,im])/(4*h);
    
    # Time step the fluid
    def fluid(self, u, ff):    
        uu = np.zeros(np.shape(u), dtype=np.complex)
        uuu = np.zeros(np.shape(u), dtype=np.complex)
        
        a, dt, rho, mu = self.a, self.dt, self.rho, self.mu
        w=u-(dt/2)*self.skew(u)+(dt/(2*rho))*ff; # Get RHS
        w=np.fft.fft(w, axis=1)
        w=np.fft.fft(w, axis=2)
        uu[0]=a[0,0]*w[0]+a[0,1]*w[1]; # Solve for LHS
        uu[1]=a[1,0]*w[0]+a[1,1]*w[1];
        uu=np.fft.ifft(uu,axis=2);
        uu=np.fft.ifft(uu,axis=1).real; #Get u at midpoint in time
        
        
        w=u-dt*self.skew(uu)+(dt/rho)*ff+(dt/2)*(mu/rho)*self.laplacian(u);# Get RHS
        w=np.fft.fft(w, axis=1)
        w=np.fft.fft(w, axis=2)
        uuu[0]=a[0,0]*w[0]+a[0,1]*w[1];# Solve for LHS
        uuu[1]=a[1,0]*w[0]+a[1,1]*w[1];
       
        uuu=np.fft.ifft(uuu,axis=2);
        uuu=np.fft.ifft(uuu,axis=1).real; # Get u at next timestep
        return uuu, uu
   

  #### Methods for plotting and visualization
    
    @property
    def vorticity(self):
        u, h, ip, im = self.u, self.h, self.ip, self.im
        return (u[1,ip,:]-u[1,im,:]-u[0,:,ip]+u[0,:,im])/(2*h);
        
    def show_vorticity(self):
        vorticity=self.vorticity
        dvorticity=(np.max(vorticity)-np.min(vorticity))/5;
        dvorticity = max(dvorticity, 0.1)  ## Catch error on 0 (or uniform) vorticity
        plt.imshow(vorticity,  vmin=-10*dvorticity, vmax=10*dvorticity, origin='lower', extent=[0, self.L, 0, self.L])
        plt.colorbar()
        
    def show_streamlines(self, cmap=None):
        X, Y = np.meshgrid(self.h*np.arange(self.N), self.h*np.arange(self.N))
        if cmap is None:
            plt.streamplot(X, Y, self.u[0], self.u[1], color='black')
        else:
            uu = np.sqrt(np.sum(self.u**2, axis=0))
            plt.streamplot(X, Y, self.u[0], self.u[1], color=uu, cmap=cmap)
            plt.colorbar()
            

        
        
