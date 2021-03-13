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
def CIRCLE(Nb, RAD, POS):  
    theta = np.linspace(0, 2*np.pi, Nb+1)[:-1]
    return np.array(POS)[None, :] + RAD*np.vstack([np.cos(theta), np.sin(theta)]).transpose()

def FULL_CIRCLE(h, RAD, POS, phi=0*np.pi/3):
    pts = []
    for y in np.arange(-RAD+h, RAD, h):
        for x in np.arange(0, (RAD**2-y**2)**0.5, h):
            pts.append([x, y])
            if x != 0:
                pts.append([-x, y])
    X, Y = np.array(pts)[:,0], np.array(pts)[:,1]
    X, Y = [X*np.cos(phi) - Y*np.sin(phi), X*np.sin(phi) + Y*np.cos(phi)]
    return (np.vstack([X, Y]) + np.array(POS)[:, None]).transpose()

#     return (np.vstack([X*np.cos(phi) - Y*np.sin(phi), X*np.sin(phi) + Y*np.cos(phi)]) + np.array(POS)[:, None]).transpose()



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
    dvorticity = max(dvorticity, 0.1)  ## Catch error on 0 (or uniform) vorticity
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



        
        
