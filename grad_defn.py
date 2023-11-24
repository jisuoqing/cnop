import numpy as np
import logging


def grad_defn(process, u_pert, t, epsilon=1e-08):

    mpi_comm = process.mpi_comm
    mpi_size = process.mpi_size
    mpi_rank = process.mpi_rank

    if mpi_rank == 0:
        logging.debug("Computing gradient...")

    # compute the objective value
    if mpi_rank == 0:
        ut = process.proceed(t)
        ut_pert = process.proceed(t, u_pert=u_pert)
        j_val = - ((ut_pert - ut) ** 2).sum()

        mpi_comm.Bcast(j_val, root=0)

        ut_size = np.array([ut.size], dtype=int)
        mpi_comm.Bcast(ut_size, root=0)

        mpi_comm.Bcast(ut, root=0)
    else:
        j_val = np.empty(1, dtype=float)
        mpi_comm.Bcast(j_val, root=0)

        ut_size = np.empty(1, dtype=int)
        mpi_comm.Bcast(ut_size, root=0)

        ut = np.empty(ut_size[0], dtype=float)
        mpi_comm.Bcast(ut, root=0)

    shape = u_pert.shape

    indices_per_process = np.array_split(list(np.ndindex(shape)), mpi_size)
    my_indices = indices_per_process[mpi_rank]

    g_local = np.zeros(shape)
    for index in my_indices:
        g_local[index] = compute_g(mpi_rank, index, u_pert, epsilon, t, ut, j_val, process)

    # gather all the gradients
    g_global = np.zeros(shape)
    mpi_comm.Allreduce(g_local, g_global, op=process.mpi.SUM)

    return g_global


def compute_g(fork_id, index, u_pert, epsilon, t, ut, j_val, process):
    u_pert_eps = u_pert.copy()
    u_pert_eps[index] += epsilon
    logging.debug("Computing gradient for index {} at fork {}".format(index, fork_id))
    ut_pert_eps = process.proceed(t, u_pert=u_pert_eps, fork_id=fork_id)
    j_pert = -((ut_pert_eps - ut) ** 2).sum()
    return (j_pert - j_val) / epsilon
