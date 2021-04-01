#### Utility Functions for IBM Simulations ####
import numpy as np
import matplotlib.pyplot as plt

#### Iterate fluids and immersed solids using built-in functions
def iterate(fluid, solids):
    ff = 0. 
    ## Force density on fluid
    for solid in solids:
        solid.step_XX(fluid.u)
        ff += solid.ff         # Force at midpoint
#     ff += fluid.ff             # External force on fluid. NOTE: add this after solid.ff to keep numpy happy
    uu=fluid.step_u(ff)        # Step Fluid Velocity
    for solid in solids:
        solid.step_X(uu)       # full step using midpoint velocity    

        
        
#### Geometry
def CIRCLE(RAD=1., POS=(0.,0.), Nb=400):  
    theta = np.linspace(0, 2*np.pi, Nb+1)[:-1]
    return RAD*np.stack([np.cos(theta), np.sin(theta)], axis=1) + POS

def FULL_CIRCLE(RAD=1., POS=(0.,0.), n=200, phi=0*np.pi/3):
    nbox = 2*int(np.sqrt(n/np.pi))
    X = np.meshgrid(np.linspace(-1., 1., nbox), np.linspace(-1., 1., nbox))
    INSIDE = (X[0]**2+X[1]**2)<=1.
    return RAD*np.stack([X[0][INSIDE], X[1][INSIDE]], axis=1)+POS

def SUNFLOWER(RAD=1., POS=(0.,0.), n=200, alpha=2): 
    b = round(alpha*np.sqrt(n))      # number of boundary points
    phi = (np.sqrt(5)+1)/2           # golden ratio
    k = np.arange(n)
    r = np.sqrt( (2*k+1)/(2*n-b+1) )
    r[k>n-b] = 1
    theta = 2*np.pi*k/phi**2
    return RAD*np.stack([r*np.cos(theta), r*np.sin(theta)], axis=1) + POS#np.array(POS)[np.newaxis, :] 




#### Force Functions
def TRAPPING_PLANE(Y, L):
    out = 0*Y
    out[:,0] = 0
    out[:,1] = np.sin(2*np.pi*Y[:,1]/L)
    return out

def GRAV(solid, Y, g=980., theta=0):
    F = np.zeros(np.shape(Y))
    F[:,0] -= g*solid.M*np.sin(theta)
    F[:,1] -= g*solid.M*np.cos(theta)
    return F

def DRAG(solid, Y, cd=1.): return -cd*solid.V

def POINT_SPRING(Y, Y0, K=1.):  return -K*(Y-Y0.reshape(1, -1))







def vorticity(u, L=1.):
    N = np.shape(u)[1]
    ii = np.arange(N)
    ip = (ii+1)%N
    im = ii-1
    vorticity=(u[1][np.meshgrid(ip, ii)]
                   -u[1][np.meshgrid(im,ii)]
                   -u[0][np.meshgrid(ii,ip)]
                   +u[0][np.meshgrid(ii,im)])/(2*N/L);
    return vorticity

#### Display vorticity on a specified axis using ax.imshow(). Returns the artist. 
def show_vorticity(u, L, ax): 
    vort=vorticity(u, L)
    dvorticity=(np.max(vort)-np.min(vort))/5;
    dvorticity = max(dvorticity, 1e-6)  ## Catch error on 0 (or uniform) vorticity
    return ax.imshow(vort,  vmin=-2*dvorticity, vmax=2*dvorticity, origin='lower', extent=[0, L, 0, L])
#         plt.colorbar()




#### Plot streamlines on a specified axis using ax.streamplot(). Returns the artist. 
def show_streamlines(u, L, ax, cmap=None,):
    N = np.shape(u)[1]
    X, Y = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))
    if cmap is None:
        return ax.streamplot(X, Y, u[0], u[1], color='black')
    else:
        uu = np.sqrt(np.sum(u**2, axis=0))
        return ax.streamplot(X, Y, u[0], u[1], color=uu.transpose(), cmap=cmap)



        
        
