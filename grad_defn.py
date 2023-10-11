import numpy as np
from multiprocessing import Pool


def grad_defn(process, u_pert, t, epsilon=1e-08, nprocs=1):
    print("Computing gradient...")
    # compute the objective value
    ut = process.proceed(t)
    ut_pert = process.proceed(t, u_pert=u_pert)
    j_val = - ((ut_pert - ut) ** 2).sum()

    with Pool(processes=nprocs) as pool:
        # Create a list of indices for each element in the 'u_pert' array
        all_index = []
        for index, _ in np.ndenumerate(u_pert):
            all_index.append(index)

        # Compute g in parallel
        results = pool.starmap(compute_g,
                               [(fork_id, index, u_pert, epsilon, t, ut, j_val, process)
                                for fork_id, index in enumerate(all_index)])

    # Process the results and update 'g'
    g = np.zeros(u_pert.shape)
    for index, value in results:
        g[index] = value

    return g


def compute_g(fork_id, index, u_pert, epsilon, t, ut, j_val, process):
    u_pert_eps = u_pert.copy()
    u_pert_eps[index] += epsilon
    print("Computing gradient for index {} at fork {}".format(index, fork_id))
    ut_pert_eps = process.proceed(t, u_pert=u_pert_eps, fork_id=fork_id)
    j_pert = -((ut_pert_eps - ut) ** 2).sum()
    return index, (j_pert - j_val) / epsilon
