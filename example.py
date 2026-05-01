#### Test: A bunch of boundaries in a shear flow ####
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# plt.rcParams['animation.writer'] = animation.writers['pillow']
ani_path = 'tutorial_figures/'


from src.ib2 import IB2
from src.fluid2 import FLUID    #### Generic fluid solver
from src.ibsim import IBSIM
from src.util import CIRCLE 

import warnings; warnings.simplefilter('ignore')

def setupShearSim():
    def shear(u):
        u[0][:, 0] = 1.
        u[0][:,-1] = -1.
        u[1][:, 0] = 0.
        u[1][:,-1] = 0.


    fluid = FLUID(L=2.)
    fluid.boundary = shear
    for i in range(1000): uu = fluid.step_u(0.)  ## spin-up: Let fluid run before adding objects to get shear flow started
    print('Fluid Spun Up')


    circle1 = CIRCLE( 0.18, (0.1, 0.25), 100)*fluid.L
    circle2 = CIRCLE( 0.18, (0.5, 0.5),  100)*fluid.L
    circle3 = CIRCLE( 0.18, (0.9, 0.75), 100)*fluid.L

    sim = IBSIM(fluid)
    for circle in [circle1, circle2, circle3]: sim.add_solid(circle)
    return sim
    
    
    

if __name__ == "__main__":
    
    ## Setup and run simulation
    sim = setupShearSim()
    Xarr, uarr = sim.run(700, fl_ufunc = lambda fluid: fluid.vorticity)   


    fluid = FLUID(L=2.)
    

    ## Animation
    fig, ax = plt.subplots()
    fig.figsize=(12, 8)
    plt.xlim([0, fluid.L])
    plt.ylim([0, fluid.L])

    cmap = plt.get_cmap('tab10')

    ims = []
    def plot_frame(i):
        im = [ax.imshow(uarr[i], origin='lower', extent=[0, fluid.L, 0, fluid.L], vmin=0, vmax=5)]
    #     im.append(fluid.show_streamlines())
        for j, _X in enumerate(Xarr[i]):
            im.append(ax.scatter(_X[:,0]%fluid.L, _X[:,1]%fluid.L, color=cmap(2*j)))
            im.append(ax.scatter([_X[0,0]%fluid.L], [_X[0, 1]%fluid.L], color=cmap(2*j+1)))
        im.append(ax.text(0, 2.1, 't={:.2f}'.format(i*fluid.dt*10)))
        return im

    ims = [plot_frame(i) for i in range(len(Xarr))]
    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=100)
    ani.save(ani_path+'circle_in_init_shear.gif')        

