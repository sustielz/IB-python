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



############## Trisph functionality  ###############
import matplotlib.tri as tri

def dodec():    #constructs the vertices of a dodecahedron on the unit sphere
    theta=2*np.pi/5
    z=np.cos(theta)/(1-np.cos(theta))
    r=np.sqrt(1-z**2)
    
    x = np.zeros([12, 3])
    x[0]=[0,0, 1];
    x[11]=[0,0,-1];
    for j in np.arange(5)+1:
        k = j-.5
        x[j]=[r*np.cos(j*theta),r*np.sin(j*theta), z]
        x[5+j]=[r*np.cos(k*theta),r*np.sin(k*theta),-z]
    
    v = np.vstack([[1,2,3], [1,3,4], [1,4,5], [1,5,6], [1,6,2],
                      [2,8,3], [3,9,4], [4,10,5], [5,11,6], [6,7,2],
                      [8,2,7], [9,3,8], [10,4,9], [11,5,10], [7,6,11],
                      [7,12,8], [8,12,9], [9,12,10], [10,12,11], [11,12,7]]) - 1
    
    return x, np.array(v, dtype=int)
    
def refine(x, v):  # script to refine triangulation of the sphere
    nv = np.shape(x)[0]
    nt = np.shape(v)[0]*1
    v = np.append(v, np.zeros([3*nt, 3], dtype=int), axis=0) ## we will have 4x as many triangles after refinement        
    x = np.append(x, np.zeros([3*nv, 3]), axis=0) ## we will have 4x as many triangles after refinement    

    next3=[1,2,0]
    vnew = np.zeros(3)
    a=np.zeros([nv+1,nv+1])
    for t in range(nt):
        for j in range(3):
            v1=v[t, next3[j]]
            v2=v[t,next3[next3[j]]]
            if a[v1,v2]==0:
                vnew[j]=nv
                x[nv]= locate(v1, v2, x)
                a[v1,v2]=nv
                a[v2,v1]=nv
                nv += 1

            else:
                vnew[j]=a[v1,v2];


        v[1*nt+t]=[v[t,0],vnew[2],vnew[1]]
        v[2*nt+t]=[v[t,1],vnew[0],vnew[2]]
        v[3*nt+t]=[v[t,2],vnew[1],vnew[0]]
        v[t]=vnew

    return x[:nv], v
    
def trisph(rad=1., ctr=[0,0,0], numr=0):
    x, v = dodec()
    for nr in range(numr):
        x, v = refine(x, v)
    
    x=x*rad+ctr
    return x, v

    
def locate(v1,v2,x):
    #locate the perpendicular bisector of the great circle joining
    #the points whose indices are stored in rows v1 and v2 of x
    xout=0.5*(x[v1]+x[v2])
    r=np.linalg.norm(xout, axis=0)
    return xout/r

        
