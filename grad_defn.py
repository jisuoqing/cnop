import numpy as np
import logging
import pathlib
import time
from utils import print_progress, wait_for_files


def grad_defn(process, u_pert, t, epsilon, restart=False, iter0=None):
    mpi_comm = process.mpi_comm
    mpi_size = process.mpi_size
    mpi_rank = process.mpi_rank
    mpi_root_dir = process.mpi_root_dir

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
    g_local = np.zeros(shape)

    indices_to_be_computed = list(np.ndindex(shape))

    if restart:
        # record the indices that have been computed in the last iteration and prepare to compute the rest
        for index in indices_to_be_computed.copy():
            tmp_fn = "{}/tmp_grad_defn_iter_{}_index_{}_{}_{}.npy".format(mpi_root_dir, iter0, index[0], index[1], index[2])
            if pathlib.Path(tmp_fn).exists():
                if mpi_rank == 0:
                    # loading is done by rank 0 only to avoid duplicated counting,
                    # and then broadcasted to all ranks by Allreduce at the end
                    g_local[tuple(index)] = np.load(tmp_fn)
                    logging.debug("Rank {}: Loaded gradient for index {}".format(mpi_rank, index))
                indices_to_be_computed.remove(index)
        if mpi_rank == 0 and (g_local.size - len(indices_to_be_computed)) > 0:
            print(f"Rank {mpi_rank}: {g_local.size - len(indices_to_be_computed)} indices already computed and loaded; "
                  f"{len(indices_to_be_computed)} indices to be computed")

    mpi_comm.Barrier()

    indices_per_process = np.array_split(indices_to_be_computed, mpi_size)
    my_indices = indices_per_process[mpi_rank]

    time_elapsed = np.empty(len(my_indices), dtype=float)
    for i, index in enumerate(my_indices):
        logging.debug(
            "Rank {}: Computing gradient [{}/{}] for index {}".format(mpi_rank, i + 1, len(my_indices), index))
        time_start = time.time()

        # create tmp files for grad_defn restart
        tmp_fn = "{}/tmp_grad_defn_iter_{}_index_{}_{}_{}.npy".format(mpi_root_dir, iter0, index[0], index[1], index[2])
        g_local[tuple(index)] = compute_g(mpi_rank, index, u_pert, epsilon, t, ut, j_val, process)
        np.save(tmp_fn, g_local[tuple(index)])

        time_end = time.time()
        logging.debug("Rank {}: Gradient [{}/{}] for index {} is {}".format(mpi_rank, i + 1, len(my_indices), index,
                                                                            g_local[tuple(index)]))
        time_elapsed[i] = time_end - time_start

        if mpi_rank == 0:
            print_progress(f"Finished [{i + 1}/{len(my_indices)}] at rank {mpi_rank} in {time_elapsed[i]:.2f} s, "
                           f"min: {time_elapsed[:i+1].min():.2f} s, max: {time_elapsed[:i+1].max():.2f} s")

    # get the list of last tmp file name for each rank to check if all ranks have finished computing gradient
    tmp_fns_last = []
    for each_rank in range(mpi_size):
        if len(indices_per_process[each_rank]) > 0:  # if there are indices to be computed by this rank
            tmp_fns_last.append("{}/tmp_grad_defn_iter_{}_index_{}_{}_{}.npy".
                                format(mpi_root_dir, iter0, *indices_per_process[each_rank][-1]))
    if not wait_for_files(tmp_fns_last, timeout=60, poll_interval=1):
        logging.error("Rank {}: Not all ranks finished computing gradient".format(mpi_rank))
        mpi_comm.Abort()
        process.mpi.Finalize()

    # get the maximum and minimum of time_elapsed
    if not restart:
        time_min = np.zeros(2, dtype=float)
        time_max = np.zeros(2, dtype=float)
        mpi_comm.Reduce(np.array([time_elapsed.min(), time_elapsed.sum()]), time_min, op=process.mpi.MIN)
        mpi_comm.Reduce(np.array([time_elapsed.max(), time_elapsed.sum()]), time_max, op=process.mpi.MAX)
        if mpi_rank == 0:
            print('\x1b[1A')  # reset the line due to print_progress
            print("Computing time across all ranks: \n"
                  "Single run: min = {}, max = {}, \n"
                  "Total  run: min = {}, max = {}".format(time_min[0], time_max[0], time_min[1], time_max[1]))

    # gather all the gradients
    g_global = np.zeros(shape)
    logging.debug("Rank {}: Gathering gradients...".format(mpi_rank))
    mpi_comm.Allreduce(g_local, g_global, op=process.mpi.SUM)
    logging.debug("Rank {}: Gradients gathered".format(mpi_rank))

    # At these stage, all ranks have computed/loaded the gradients, so we can delete the tmp files for iter0
    if mpi_rank == 0:
        for index in list(np.ndindex(shape)):
            tmp_fn = "{}/tmp_grad_defn_iter_{}_index_{}_{}_{}.npy".format(mpi_root_dir, iter0,
                                                                          index[0], index[1], index[2])
            if pathlib.Path(tmp_fn).exists():
                pathlib.Path(tmp_fn).unlink()

    return g_global


def compute_g(fork_id, index, u_pert, epsilon, t, ut, j_val, process):
    u_pert_eps = u_pert.copy()
    u_pert_eps[tuple(index)] += epsilon
    ut_pert_eps = process.proceed(t, u_pert=u_pert_eps, fork_id=fork_id)
    j_pert = -((ut_pert_eps - ut) ** 2).sum()
    return (j_pert - j_val) / epsilon
