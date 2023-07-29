import numpy as np
from utils import compute_obj


def grad_defn(process, u_pert, t, epsilon=1e-08):

    # compute the objective value
    ut = process.proceed(t)
    ut_pert = process.proceed(t, u_pert=u_pert)
    j_val = - ((ut_pert - ut) ** 2).sum()

    # compute the gradient
    g = np.zeros(u_pert.shape)

    for index, value in np.ndenumerate(u_pert):
        u_pert_eps = u_pert.copy()
        u_pert_eps[index] += epsilon
        ut_pert_eps = process.proceed(t, u_pert=u_pert_eps)
        j_pert = - ((ut_pert_eps - ut)**2.).sum()
        g[index] = (j_pert - j_val) / epsilon

    return g
