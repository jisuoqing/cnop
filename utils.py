import numpy as np


def do_projection(u, delta=8e-4):
    sum0 = (u**2.).sum()
    if np.sqrt(sum0) <= delta:
        proj_u = u
    else:
        proj_u = delta / np.sqrt(sum0) * u
    return proj_u


def usphere_sample(n):
    # Generate standard normal random variables
    tmp = np.random.randn(n)
    # Find the magnitude of each column.
    # Square each element, add and take the square root.

    mag = np.sqrt((tmp**2.).sum())

    # Make a diagonal matrix of them -- inverses.
    dm = 1.0 / mag
    # Multiply to scale properly.
    # Transpose so x contains the observations.
    x = dm * tmp
    return x


def compute_obj(process, u_pert, t):
    # compute the objective value
    ut = process.proceed(t)
    ut_pert = process.proceed(t, u_pert=u_pert)
    j_val = - ((ut_pert - ut) ** 2).sum()
    return j_val
