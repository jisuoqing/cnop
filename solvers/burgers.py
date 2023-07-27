import numpy as np


class Burgers:
    def __init__(self, u_init, t0, **kwargs):
        # for given initial condition u_init, evolve it to time t0 as the basic state u0
        self.u_init = u_init
        self.t0 = t0
        self.vis = kwargs.get('vis', 0.5)
        self.delta_t = kwargs.get('delta_t', 0.1)
        self.delta_x = kwargs.get('delta_x', 1.0)
        self.x = np.arange(0, len(self.u_init)) * self.delta_x
        nt0 = int(self.t0 / self.delta_t + 1)
        # now evolve the initial condition to t0, to obtain u0 which is the basic state
        import importlib
        self.solve = getattr(importlib.import_module("solvers.burgers_lib"), "solve_burgers")
        self.u0 = self.solve(u_init, nt0, self.vis, self.delta_t, self.delta_x)
        # now print that the class is initialized with detailed information
        print("The basic state is evolved from the initial condition to time {}.".format(self.t0))
        return

    def proceed(self, t1, u_pert=None):
        nt = int(t1 / self.delta_t + 1)
        if u_pert is None:
            ut = self.solve(self.u0, nt, self.vis, self.delta_t, self.delta_x)
        else:
            # perturbation is added at time t0 and then evolved over another t1
            ut = self.solve(self.u0 + u_pert, nt, self.vis, self.delta_t, self.delta_x)
        return ut
