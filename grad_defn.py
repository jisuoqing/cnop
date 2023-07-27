import numpy as np
from utils import compute_obj


def grad_defn(process, u_pert, t, epsilon=1e-08):

    # compute the objective value
    ut = process.proceed(t)
    ut_pert = process.proceed(t, u_pert=u_pert)
    j_val = - ((ut_pert - ut) ** 2).sum()

    # compute the gradient
    g = np.zeros(process.u0.shape)
    g[0] = 0.
    g[-1] = 0.

    for i in np.arange(1, len(process.u0) - 1):
        u_pert_eps = u_pert.copy()
        u_pert_eps[i] += epsilon
        ut_pert_eps = process.proceed(t, u_pert=u_pert_eps)
        j_pert = - ((ut_pert_eps - ut)**2.).sum()
        g[i] = (j_pert - j_val) / epsilon

    return g
