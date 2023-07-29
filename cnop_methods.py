import numpy as np


def spg2_defn(process, u_pert, t1):
    from utils import do_projection, compute_obj
    from grad_defn import grad_defn

    iter0 = 0

    max_float = 1.e100  # np.finfo(float).max
    min_float = 1.e-100  # np.finfo(float).tiny

    max_iter = 300
    ifcnt = 0

    max_ifcnt = 100000
    igcnt = 0

    eps = 1e-8
    gamma = 0.0001

    # storage M = 10 recent numbers
    j_num = 10
    j_values = -np.inf * np.ones(j_num)
    u_pert = do_projection(u_pert)
    u_pert_best = u_pert.copy()

    # compute objective value
    j_val = compute_obj(process, u_pert, t1)
    j_values[0] = j_val
    j_best = j_val
    ifcnt += 1

    # compute gradient (adjoint method)
    g = grad_defn(process, u_pert, t1, epsilon=1e-08)
    igcnt += 1

    # step-1: discriminate whether the current point is stationary
    cg = u_pert - g
    cg = do_projection(cg)
    cgnorm = (np.abs(cg - u_pert)).max()

    if cgnorm != 0:
        lambda_ = 1 / cgnorm

    # step-2:   Backtracking
    while cgnorm > eps and iter0 <= max_iter and ifcnt <= max_ifcnt:
        iter0 += 1
        print("----------------------- iter", iter0, "-----------------------")

        # step-2.1: compute d
        d = u_pert - lambda_ * g
        d = do_projection(d)
        d = d - u_pert
        gtd = (g*d).sum()

        # step-2.2 and step 2.3: compute alpha (lambda in paper) and u0_new,
        j_max = j_values.max()
        u_pert_new = u_pert + d
        j_new = compute_obj(process, u_pert_new, t1)
        ifcnt = ifcnt + 1
        alpha = 1

        while j_new > j_max + gamma * alpha * gtd:
            if alpha <= 0.1:
                alpha = alpha / 2.
            else:
                atemp = - gtd * alpha ** 2 / (2 * (j_new - j_val - alpha * gtd))
                if atemp < 0.1 or atemp > 0.9 * alpha:
                    atemp = alpha / 2.
                alpha = atemp
            u_pert_new = u_pert + alpha * d
            j_new = compute_obj(process, u_pert_new, t1)
            ifcnt += 1

        j_val = j_new
        j_values[np.mod(iter0, j_num)] = j_val  # store the recent j_num values
        if j_new < j_best:
            j_best = j_new
            u_pert_best = u_pert_new.copy()
        g_new = grad_defn(process, u_pert_new, t1)
        igcnt = igcnt + 1

        # step-3: compute lambda (alpha in paper)
        s = u_pert_new - u_pert
        y = g_new - g
        sts = (s**2.).sum()
        sty = (s*y).sum()
        u_pert = u_pert_new.copy()
        g = g_new.copy()
        cg = u_pert - g
        cg = do_projection(cg)
        cgnorm = (np.abs(cg - u_pert)).max()

        if sty <= 0:
            lambda_ = max_float
        else:
            lambda_ = np.min((max_float, np.max((min_float, sts / sty))))

        print("lambda = ", lambda_)
        print("j_val = ", j_val)
        print("sts = ", sts)
        print("sty = ", sty)
        print("cgnorm = ", cgnorm)

    if cgnorm <= eps:
        print('convergence')
    else:
        if iter0 > max_iter:
            print('too many iterations')
        else:
            if ifcnt > max_ifcnt:
                print('too many function evaluations')
            else:
                print('unknown stop')

    return u_pert_best, j_best
