import numpy as np
import logging
import time
from utils import print_progress


def grad_defn(process, u_pert, t, epsilon):
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
    time_elapsed = np.empty(len(my_indices), dtype=float)
    for i, index in enumerate(my_indices):
        if mpi_rank == 0:
            print_progress(f"Computing gradient [{i + 1}/{len(my_indices)}] at rank {mpi_rank}...")

        logging.debug(
            "Rank {}: Computing gradient [{}/{}] for index {}".format(mpi_rank, i + 1, len(my_indices), index))
        time_start = time.time()
        g_local[tuple(index)] = compute_g(mpi_rank, index, u_pert, epsilon, t, ut, j_val, process)
        time_end = time.time()
        logging.debug("Rank {}: Gradient [{}/{}] for index {} is {}".format(mpi_rank, i + 1, len(my_indices), index,
                                                                            g_local[tuple(index)]))
        time_elapsed[i] = time_end - time_start

    # get the maximum and minimum of time_elapsed
    time_min = np.zeros(2, dtype=float)
    time_max = np.zeros(2, dtype=float)
    mpi_comm.Reduce(np.array([time_elapsed.min(), time_elapsed.sum()]), time_min, op=process.mpi.MIN)
    mpi_comm.Reduce(np.array([time_elapsed.max(), time_elapsed.sum()]), time_max, op=process.mpi.MAX)
    if mpi_rank == 0:
        print("Gradient computation time: \n"
              "Per run:  min = {}, max = {}, \n"
              "Per core: min = {}, max = {}".format(time_min[0], time_max[0], time_min[1], time_max[1]))
    # gather all the gradients
    g_global = np.zeros(shape)
    logging.debug("Rank {}: Gathering gradients...".format(mpi_rank))
    mpi_comm.Allreduce(g_local, g_global, op=process.mpi.SUM)
    logging.debug("Rank {}: Gradients gathered".format(mpi_rank))

    return g_global


def compute_g(fork_id, index, u_pert, epsilon, t, ut, j_val, process):
    u_pert_eps = u_pert.copy()
    u_pert_eps[tuple(index)] += epsilon
    ut_pert_eps = process.proceed(t, u_pert=u_pert_eps, fork_id=fork_id)
    j_pert = -((ut_pert_eps - ut) ** 2).sum()
    return (j_pert - j_val) / epsilon
