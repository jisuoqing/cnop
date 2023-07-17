import numpy as np
from solvers import solve


def grad_defn(u0, u_pert, t, vis, delta_t, delta_x, epsilon=1e-08):

    # compute the objective value
    ut = solve(u0, t, vis, delta_t, delta_x)
    ut_pert = solve(u0 + u_pert, t, vis, delta_t, delta_x)
    j_val = - ((ut_pert - ut) ** 2).sum()

    # compute the gradient
    g = np.zeros(u0.shape)
    g[0] = 0.
    g[-1] = 0.

    for i in np.arange(1, len(u0) - 1):
        init = u0 + u_pert
        init[i] += epsilon
        ut_pert_eps = solve(init, t, vis, delta_t, delta_x)
        j_pert = - ((ut_pert_eps - ut)**2.).sum()
        g[i] = (j_pert - j_val) / epsilon

    return g
