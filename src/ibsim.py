## Top-level class for an IB simulation. Add your fluids and solids, and then run. 
from .ib2 import IB2
class IBSIM(object):
    def __init__(self, fluid, solids=[]):
        self.fluid = fluid
        self.solids = solids

    #### Add solid of a particular type, geometry, etc
    def add_solid(self, X, IB=IB2, **kwargs):
        self.solids.append(IB(X, self.fluid.N, self.fluid.h, self.fluid.dt, **kwargs))

    #### Iterate fluids and immersed solids using built-in functions. Option for external force.
    def iterate(self, ff_ext=0.):
        ff = 0.                                    ## Force density on fluid
        for solid in self.solids:                  ## Half-step position
            solid.step_XX(self.fluid.u)                                   
            ff += solid.ff 
        ff += ff_ext                         ##  **NOTE: add this after the solid forces (not before), otherwise numpy gets cranky 
        uu=self.fluid.step_u(ff)                    ## Step Fluid Velocity
        for solid in self.solids: solid.step_X(uu)  ## Full-step position

    ## Run for nsteps. Write output once per nmod frames for solids, and once per nmod*nmod_fluid frames for fluids.
    ##   -  The default output is boundary position and fluid velocity, but ib_ufunc and fl_ufunc can be modified to 
    ##   -  output other properties, or even plot figures
    def run(self, nsteps, nmod=10, nmod_fluid=1, ib_ufunc = lambda solid: solid.X*1., fl_ufunc = lambda fluid: fluid.u*1.):
        ib_out = []
        fl_out = []
        for i in range(nsteps):
            self.iterate()
            if i % nmod == 0: 
                print(f"step {i}...")
                ib_out.append([ib_ufunc(solid) for solid in self.solids])
            if i % nmod*nmod_fluid == 0: 
                fl_out.append(fl_ufunc(self.fluid))
        return ib_out, fl_out

    
