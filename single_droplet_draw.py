### General Setup
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import json
import sys


#######    Plotting Functions      ##############
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
#################################################

        
if len(sys.argv)==1:
    PARAMNAME = 'params'
    DATANAME = 'data'
    IM1NAME = 'stability'
    IM2NAME = 'single_droplet'
else:
    FILENAME = sys.argv[1]
    PARAMNAME = 'params/'+FILENAME
    DATANAME = 'data/'+FILENAME
    IM1NAME = 'ims/'+FILENAME
    IM2NAME = 'ims/'+FILENAME

with open(PARAMNAME+'.json', 'r') as f:
    params = json.load(f)
data = np.load(DATANAME+'.npz')
params.update(data)
locals().update(params)





delta = [[np.max(np.linalg.norm(Y[j][i] - Xin[j][i], axis=1)) for i in range(len(Y[0]))] for j in range(len(Y))] 

fig_delta = plt.figure()
ax_delta = fig_delta.add_subplot(111)
ax_delta.plot(np.array(delta[0])*N/L)
ax_delta.set_ylim([0, 3])
ax_delta.set_ylabel('|Y-X|/h')
ax_delta.set_xlabel('timestep')
ax_delta.set_title('Stability')
fig_delta.savefig(IM1NAME+'.png')


#### Animation
fig = plt.figure(constrained_layout=True)
nfigs = len(Xin)
gs = fig.add_gridspec(nfigs, nfigs+1)
ax = fig.add_subplot(gs[:, :-1])
ax.set_xlim([0, L])
ax.set_ylim([0, L])

axes_frame = [fig.add_subplot(gs[i, -1]) for i in range(nfigs)]
for j, axj in enumerate(axes_frame):
#     axj.set_title('$K={}, M={}, N_i={}$'.format(outsides[j].K, insides[j].M, len(insides[j].X)))
    axj.set_title('$K={}, M={}, N_i={}$'.format(K, M, Nb))
    
#     axj.set_xlim([-1, 1])
#     axj.set_ylim([-1, 1])
cmap = plt.get_cmap('tab10')
RED = plt.get_cmap('Reds')
ims = []

for i, u in enumerate(U):
    ii = i*nmod
    im = [show_vorticity(u, L, ax)]
    out = show_streamlines(u, L, ax)
    im.append(out.lines)
    for k, X in enumerate([Xin, Xout]):
        for j, x in enumerate(X): 
            im.append(ax.scatter(x[ii][:,0]%L, x[ii][:,1]%L, color=cmap(2*j+k)))
            im.append(ax.text(0.5, 1.01, 'Time {}'.format(i*nmod*dt), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes))

    for j, axj in enumerate(axes_frame):
        #### Plot Droplets in COM Frame
        xin = Xin[j][ii]
        xout = Xout[j][ii]
        com = np.mean(xin, axis=0)
        ins = xin - com
        out = xout - com

        im.append(axj.scatter(ins[:,0], ins[:,1], s=1000/Ni, color=cmap(0)))
        im.append(axj.scatter(out[:,0], out[:,1], color=cmap(1)))
        im.append(axj.scatter([out[0,0]], [out[0,1]], color='red'))  ## Mark theta=0

        #### DEBUG ####
#         yins = Y[j][i] - com
#         im.append(axj.scatter(yins[:,0], yins[:,1], s=1000/Ni, color=cmap(1)))
#         TETHERS = np.array([[ins[i], yins[i]] for i in range(len(ins))])
#         for tether in TETHERS:
#             im.extend(axj.plot(tether[:,0], tether[:,1], color=RED(np.linalg.norm(tether[1]-tether[0])*N/L)))

        ## Record theta profile of each boundary
#             THETA[j].append(np.arctan2(out[:,1], out[:,0]))


    ims.append(im) 
        
#### Credit: Stack Exchange  https://stackoverflow.com/questions/61932534/cannot-remove-streamplot-arrow-heads-from-matplotlib-axes
from matplotlib import patches
for art in ax.get_children():
    if not isinstance(art, patches.FancyArrowPatch):
        continue
    art.remove()        # Method 1
    # art.set_alpha(0)  # Method 2  


    
    
ani2 = animation.ArtistAnimation(fig, ims, interval=150, repeat_delay=1)
ani2.save(IM2NAME+'.gif', writer='pillow')


