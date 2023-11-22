import numpy as np
import logging
from mpi4py import MPI


def grad_defn(process, u_pert, t, epsilon=1e-08):

    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    print("Computing gradient...")
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

    print("Rank {} has {} indices".format(mpi_rank, len(my_indices)))

    g_local = np.zeros(shape)
    for index in my_indices:
        value = compute_g(mpi_rank, index, u_pert, epsilon, t, ut, j_val, process)
        g_local[index] = value

    all_g_local = mpi_comm.gather(g_local, root=0)

    # gather all the gradients
    if mpi_rank == 0:
        g_global = np.zeros(shape)
        for g in all_g_local:
            g_global += g
        mpi_comm.bcast(g_global, root=0)
    else:
        g_global = np.zeros(shape)
        mpi_comm.bcast(g_global, root=0)

    return g_global


def compute_g(fork_id, index, u_pert, epsilon, t, ut, j_val, process):
    u_pert_eps = u_pert.copy()
    u_pert_eps[index] += epsilon
    print("Computing gradient for index {} at fork {}".format(index, fork_id))
    ut_pert_eps = process.proceed(t, u_pert=u_pert_eps, fork_id=fork_id)
    j_pert = -((ut_pert_eps - ut) ** 2).sum()
    return (j_pert - j_val) / epsilon
