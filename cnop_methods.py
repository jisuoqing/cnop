import numpy as np
from sim_controller import load_checkpoint, save_checkpoint


class Spg2Defn:
    def __init__(self, process, u_pert, t1):
        from utils import do_projection, compute_obj
        from grad_defn import grad_defn

        self.mpi_comm = process.mpi_comm
        self.mpi_rank = self.mpi_comm.Get_rank()

        if process.restart:
            # load the method info from the restart checkpoint
            load_checkpoint(process.restart_checkpoint_fn, "method", self)
        else:

            self.iter0 = 0

            self.max_float = 1.e100  # np.finfo(float).max
            self.min_float = 1.e-100  # np.finfo(float).tiny

            self.max_iter = 300
            self.ifcnt = 0

            self.max_ifcnt = 100000
            self.igcnt = 0

            self.eps = 1e-8
            self.gamma = 0.0001

            # storage M = 10 recent numbers
            self.j_num = 10
            self.j_values = -np.inf * np.ones(self.j_num)
            self.u_pert = do_projection(u_pert)
            self.u_pert_best = self.u_pert.copy()

            # compute objective value
            if self.mpi_rank == 0:
                self.j_val = compute_obj(process, self.u_pert, t1)
                self.mpi_comm.Bcast(self.j_val, root=0)
            else:
                self.j_val = np.empty(1, dtype=float)
                self.mpi_comm.Bcast(self.j_val, root=0)
            self.j_values[0] = self.j_val
            self.j_best = self.j_val
            self.ifcnt += 1

            # compute gradient (adjoint method)
            self.g = grad_defn(process, self.u_pert, t1, epsilon=1e-08)
            self.igcnt += 1

            # step-1: discriminate whether the current point is stationary
            cg = self.u_pert - self.g
            cg = do_projection(cg)
            self.cgnorm = (np.abs(cg - self.u_pert)).max()

            if self.cgnorm != 0:
                self.lambda_ = 1 / self.cgnorm

            # save all needed information for restart
            if self.mpi_rank == 0:
                print("----------------------- iter", self.iter0, "-----------------------")
                print("lambda = ", self.lambda_)
                print("j_val = ", self.j_val)
                print("cgnorm = ", self.cgnorm)
                save_checkpoint(process=process, method=self)

        # step-2:   Backtracking
        while self.cgnorm > self.eps and self.iter0 <= self.max_iter and self.ifcnt <= self.max_ifcnt:
            self.iter0 += 1

            # step-2.1: compute d
            d = self.u_pert - self.lambda_ * self.g
            d = do_projection(d)
            d = d - self.u_pert
            gtd = (self.g * d).sum()

            # step-2.2 and step 2.3: compute alpha (lambda in paper) and u0_new,
            j_max = self.j_values.max()
            u_pert_new = self.u_pert + d
            if self.mpi_rank == 0:
                j_new = compute_obj(process, u_pert_new, t1)
                self.mpi_comm.Bcast(j_new, root=0)
            else:
                j_new = np.empty(1, dtype=float)
                self.mpi_comm.Bcast(j_new, root=0)
            self.ifcnt = self.ifcnt + 1
            alpha = 1

            while j_new > j_max + self.gamma * alpha * gtd:
                if alpha <= 0.1:
                    alpha = alpha / 2.
                else:
                    atemp = - gtd * alpha ** 2 / (2 * (j_new - self.j_val - alpha * gtd))
                    if atemp < 0.1 or atemp > 0.9 * alpha:
                        atemp = alpha / 2.
                    alpha = atemp
                u_pert_new = self.u_pert + alpha * d
                if self.mpi_rank == 0:
                    j_new = compute_obj(process, u_pert_new, t1)
                    self.mpi_comm.Bcast(j_new, root=0)
                else:
                    j_new = np.empty(1, dtype=float)
                    self.mpi_comm.Bcast(j_new, root=0)
                self.ifcnt += 1

            self.j_val = j_new
            self.j_values[np.mod(self.iter0, self.j_num)] = self.j_val  # store the recent self.j_num values
            if j_new < self.j_best:
                self.j_best = j_new
                self.u_pert_best = u_pert_new.copy()
            g_new = grad_defn(process, u_pert_new, t1)
            self.igcnt += 1

            # step-3: compute lambda (alpha in paper)
            s = u_pert_new - self.u_pert
            y = g_new - self.g
            sts = (s ** 2.).sum()
            sty = (s * y).sum()
            self.u_pert = u_pert_new.copy()
            self.g = g_new.copy()
            cg = self.u_pert - self.g
            cg = do_projection(cg)
            self.cgnorm = (np.abs(cg - self.u_pert)).max()

            if sty <= 0:
                self.lambda_ = self.max_float
            else:
                self.lambda_ = np.min((self.max_float, np.max((self.min_float, sts / sty))))

            # save all needed information for restart
            if self.mpi_rank == 0:
                print("----------------------- iter", self.iter0, "-----------------------")
                print("lambda = ", self.lambda_)
                print("j_val = ", self.j_val)
                print("sts = ", sts)
                print("sty = ", sty)
                print("cgnorm = ", self.cgnorm)
                save_checkpoint(process=process, method=self)

            # # set MPI barrier to make sure all processes are on the same page
            # self.mpi_comm.Barrier()

        if self.mpi_rank == 0:
            if self.cgnorm <= self.eps:
                print('convergence')
            else:
                if self.iter0 > self.max_iter:
                    print('too many iterations')
                else:
                    if self.ifcnt > self.max_ifcnt:
                        print('too many function evaluations')
                    else:
                        print('unknown stop')

        return
