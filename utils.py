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


def burgers_obj(u0, u_pert, t, vis, delta_t, delta_x):
    from burgers_lib import solve_burgers
    # compute the objective value
    ut = solve_burgers(u0, t, vis, delta_t, delta_x)
    ut_pert = solve_burgers(u0 + u_pert, t, vis, delta_t, delta_x)
    j_val = - ((ut_pert - ut) ** 2).sum()
    return j_val
