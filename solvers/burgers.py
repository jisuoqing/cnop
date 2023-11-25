import numpy as np
from sim_controller import find_latest_checkpoint, load_checkpoint
from utils import usphere_sample
from mpi4py import MPI


class Burgers:
    def __init__(self, u_init, t0, **kwargs):
        self.mpi = MPI
        self.mpi_comm = self.mpi.COMM_WORLD
        self.mpi_comm_self = self.mpi.COMM_SELF
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()

        # check if there is a checkpoint file in the base_dir
        try:
            last_checkpoint_fn = find_latest_checkpoint(kwargs.get('base_dir', './'),
                                                        self.__class__.__name__ + "_checkpoint")
        except FileNotFoundError:
            self.restart = False
        else:
            self.restart = True
            self.restart_checkpoint_fn = last_checkpoint_fn

        # load the solver no matter if there is a checkpoint file
        import importlib
        self.solve = getattr(importlib.import_module("solvers.burgers_lib"), "solve_burgers")

        if self.restart:
            # if there is a checkpoint file, load process attributes from it
            load_checkpoint(self.restart_checkpoint_fn, "process", self)
            # now print that the class is initialized with detailed information
            if self.mpi_rank == 0:
                print("The class is initialized with the checkpoint file {}.".format(self.restart_checkpoint_fn))
            return
        else:
            # for given initial condition u_init, evolve it to time t0 as the basic state u0
            if not isinstance(u_init, np.ndarray):
                raise ValueError("The u_init must be a numpy array in Burgers class.")
            self.u_init = u_init
            self.t0 = t0
            self.vis = kwargs.get('vis', 0.5)
            self.delta_t = kwargs.get('delta_t', 0.1)
            self.delta_x = kwargs.get('delta_x', 1.0)
            self.x = np.arange(0, len(self.u_init)) * self.delta_x
            self.base_dir = kwargs.get('base_dir', './')
            self.basename = kwargs.get('basename', 'burgers')
            self.t1 = None
            self.ut1_unperturbed = None
            nt0 = int(self.t0 / self.delta_t + 1)
            # now evolve the initial condition to t0, to obtain u0 which is the basic state
            self.u0 = self.solve(u_init, nt0, self.vis, self.delta_t, self.delta_x)
            # now print that the class is initialized with detailed information
            if self.mpi_rank == 0:
                print("The basic state is evolved from the initial condition to time {}.".format(self.t0))
            return

    def proceed(self, t1, u_pert=None, fork_id=None):
        # fork_id is not used in this class since there is no need to create fork folders
        nt = int(t1 / self.delta_t + 1)
        if u_pert is None:
            if self.ut1_unperturbed is not None and self.t1 == t1:
                # if the unperturbed solution at t1 is already computed, then return it
                return self.ut1_unperturbed
            else:
                # otherwise, compute the unperturbed solution at t1 and return it
                ut = self.solve(self.u0, nt, self.vis, self.delta_t, self.delta_x)
                self.ut1_unperturbed = ut
                self.t1 = t1
                return ut
        else:
            # perturbation is added at time t0 and then evolved over another t1
            ut = self.solve(self.u0 + u_pert, nt, self.vis, self.delta_t, self.delta_x)
            return ut

    def generate_u_pert(self, pert_mag=1.):
        # generate a random perturbation
        if self.mpi_rank == 0:
            u_pert = np.zeros(self.u_init.shape)
            u_pert[1:-1] = usphere_sample(len(self.u_init) - 2)
            u_pert = self.mpi_comm.bcast(u_pert, root=0)
        else:
            u_pert = None
            u_pert = self.mpi_comm.bcast(u_pert, root=0)
        return u_pert * pert_mag
