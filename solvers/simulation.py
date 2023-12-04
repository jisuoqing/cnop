from sim_controller import update_parameter, find_latest_checkpoint, load_checkpoint
from utils import generate_python_wrapper, wait_for_file
import subprocess
import os
import warnings
import yt
import numpy as np
import h5py
import pathlib
import shutil
from mpi4py import MPI
import logging

yt.set_log_level("error")


class Simulation:
    def __init__(self, u_init_fn: str, base_dir: str,
                 exec_command: str, wrapper_args: str, wrapper_nproc: int,
                 wrapper_running_check_fn: str, wrapper_running_check_timeout: float,
                 wrapper_finish_check_fn: str, wrapper_finish_check_timeout: float,
                 wrapper_check_poll_interval: float,
                 init_params: dict, param_fn: str, u0_fn: str,
                 pert_var: str, grow_var: str, yt_derived_fields: callable = None,
                 link_list: list = None, copy_list: list = None):
        """
        :param u_init_fn: Initial condition file name. If None, then the initial condition is generated by the solver
            (e.g., Flash, Athena). Otherwise, the initial condition is read from the file (e.g., Gizmo)
        :param base_dir: The directory where the simulation is run
        :param exec_command: The executable command of the solver, WITHOUT "mpirun -np x" prefix
        :param wrapper_args: The arguments of the Python wrapper
        :param wrapper_nproc: The number of processors to run the Python wrapper
        :param wrapper_running_check_fn: The file name to check whether the simulation is running
        :param wrapper_running_check_timeout: The timeout to check whether the simulation is running
        :param wrapper_finish_check_fn: The file name to check whether the simulation is finished
        :param wrapper_finish_check_timeout: The timeout to check whether the simulation is finished
        :param wrapper_check_poll_interval: The interval to check whether the simulation output stops changing
        :param init_params: Selected (cnop-related) parameters to be initialized/updated in the parameter file
        :param param_fn: The parameter file name
        :param u0_fn: The file name of the basic state, which is generated by evolving the initial condition to time t0
            (where t0 MUST be specified in init_params! This is handled by the child class)
        :param pert_var: The variable name to be perturbed
        :param grow_var: The variable name whose growth rate is aimed to be maximized
        :param yt_derived_fields: The function to compute derived fields from the yt dataset, usually to compute grow_var
        :param link_list: The list of files to be linked to the fork_dir, for parallelism
        :param copy_list: The list of files to be copied to the fork_dir, for parallelism
        """

        self.mpi = MPI
        self.mpi_comm = self.mpi.COMM_WORLD
        self.mpi_comm_self = self.mpi.COMM_SELF
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()

        try:
            last_checkpoint_fn = find_latest_checkpoint(base_dir, self.__class__.__name__ + "_checkpoint")
        except FileNotFoundError:
            self.restart = False
        else:
            self.restart = True
            self.restart_checkpoint_fn = last_checkpoint_fn

        if self.restart:
            # if there is a checkpoint file, load process attributes from it
            load_checkpoint(self.restart_checkpoint_fn, "process", self)
            # derived_field function is not saved in the checkpoint file, so we need to reassign it
            self.yt_derived_fields = yt_derived_fields
            # now print that the class is initialized with detailed information
            if self.mpi_rank == 0:
                print("The class is initialized with the checkpoint file {}.".format(self.restart_checkpoint_fn))
            return
        else:
            if u_init_fn is not None:
                if not pathlib.Path(base_dir + "/" + u_init_fn).exists():
                    raise ValueError("The initial condition file does not exist!")
                self.u_init_fn = u_init_fn
            self.base_dir = base_dir
            if "./" not in exec_command:
                raise ValueError("'./' must be included before the executable!")
            self.exec_command = exec_command
            self.wrapper_args = wrapper_args
            self.wrapper_nproc = wrapper_nproc
            self.wrapper_running_check_fn = wrapper_running_check_fn
            self.wrapper_running_check_timeout = wrapper_running_check_timeout
            self.wrapper_finish_check_fn = wrapper_finish_check_fn
            self.wrapper_finish_check_timeout = wrapper_finish_check_timeout
            self.wrapper_check_poll_interval = wrapper_check_poll_interval
            self.param_fn = param_fn
            # u0_fn is the file name of the basic state, which must match the output file name of the solver
            self.u0_fn = u0_fn
            self.pert_var = pert_var
            self.grow_var = grow_var
            self.yt_derived_fields = yt_derived_fields

            self.t1 = None
            self.ut1_unperturbed_fn = None

            if link_list is None:
                link_list = []
            if copy_list is None:
                copy_list = []

            # self.u0_fn is the basic state of the not-perturbed IC, which should be *linked* (not copied to save disk
            # space)
            link_list.append(self.u0_fn)
            # self.param_fn is the parameter file, which should be *copied* (not linked to avoid overwriting)
            copy_list.append(self.param_fn)
            self.link_list = list(set(link_list))
            self.copy_list = list(set(copy_list))

            if len(self.link_list) + len(self.copy_list) != len(set(self.link_list + self.copy_list)):
                raise ValueError("The link_list and copy_list share the same item!")

            # for initializing u0, we just need to perform on one processor
            if self.mpi_rank == 0:
                if pathlib.Path(self.base_dir + "/" + self.u0_fn).exists():
                    # warnings.warn("The basic state file already exists! Deleting it now for safety.")
                    os.remove(self.base_dir + "/" + self.u0_fn)
                # Now evolve the initial condition to t0, to obtain u0 which is the basic state
                update_parameter(self.base_dir + "/" + self.param_fn, init_params)

                self.run_simulation_with_python_wrapper()

                if not pathlib.Path(self.base_dir + "/" + self.u0_fn).exists():
                    raise ValueError(
                        "The basic state file might be generated, but you might guess the file name wrong!")
                logging.debug("The basic state u0 is evolved from the initial condition u_init "
                              "and saved as {}.".format(self.u0_fn))
            # set a barrier to make sure the basic state is generated before proceeding
            self.mpi_comm.barrier()
            return

    def run_simulation_with_python_wrapper(self):
        generate_python_wrapper(self.exec_command, wrapper_path=self.base_dir)
        # Now start the simulation
        original_dir = os.getcwd()
        os.chdir(self.base_dir)
        new_comm = self.mpi_comm_self.Spawn(command='python3', args=["wrapper.py"] + self.wrapper_args.split(),
                                            maxprocs=self.wrapper_nproc)
        if self.wrapper_running_check_fn is None and self.wrapper_finish_check_fn is None:
            raise ValueError("At least one of wrapper_running_check_fn and wrapper_finish_check_fn must be specified!")
        if self.wrapper_running_check_fn is not None:
            if not wait_for_file(self.wrapper_running_check_fn,
                                 timeout=self.wrapper_running_check_timeout,
                                 poll_interval=self.wrapper_check_poll_interval):
                raise RuntimeError("The simulation is not running since "
                                   f"{self.base_dir}/{self.wrapper_running_check_fn} is not generated!")
        if self.wrapper_finish_check_fn is not None:
            if not wait_for_file(self.wrapper_finish_check_fn,
                                 timeout=self.wrapper_finish_check_timeout,
                                 poll_interval=self.wrapper_check_poll_interval):
                raise RuntimeError(f"The simulation is not finished and {self.wrapper_finish_check_fn} is not generated!"
                                   f"Please check the output {self.base_dir}/stdout.txt for more information.")
        os.chdir(original_dir)
        return

    def generate_u_pert(self, pert_mag, seed=1234):
        # generate a perturbation file with magnitude pert_mag
        # This should be created on one processor and broadcast to all processors
        if self.mpi_rank == 0:
            ds = yt.load(self.base_dir + "/" + self.u0_fn)
            np.random.seed(seed)
            u_pert = np.random.uniform(-pert_mag, pert_mag, ds.domain_dimensions)
            del ds
            self.mpi_comm.bcast(u_pert, root=0)
        else:
            u_pert = None
            u_pert = self.mpi_comm.bcast(u_pert, root=0)
        return u_pert

    def proceed_simulation(self, params, t1,
                           exec_command=None, wrapper_args=None, wrapper_nproc=None,
                           wrapper_running_check_fn=None, wrapper_running_check_timeout=None,
                           wrapper_finish_check_fn=None, wrapper_finish_check_timeout=None,
                           wrapper_check_poll_interval=None,
                           u_pert=None, u_pert_fn=None,
                           ut_fn=None, delete_fn=None,
                           fork_id=None):
        # evolve the basic state to time t1, with perturbation u_pert. t1 is specified in params
        # Note that only when we actually run the simulation, we need save u_pert into a file
        if t1 not in params.values():
            raise ValueError("The final time is not included in the input parameter!")

        old_base_dir = self.base_dir
        if fork_id is not None:
            # create a separate run in a sub folder
            fork_dir = self.base_dir + "/fork_%d" % fork_id
            self.make_fork_dir(fork_dir)
            self.base_dir = fork_dir

        if exec_command is not None:
            # if a different exe is needed for restarting
            self.exec_command = exec_command
        if wrapper_args is not None:
            # if different args is needed for restarting
            self.wrapper_args = wrapper_args
        if wrapper_nproc is not None:
            # if a different # of processors is needed for restarting
            self.wrapper_nproc = wrapper_nproc
        if wrapper_running_check_fn is not None:
            self.wrapper_running_check_fn = wrapper_running_check_fn
        if wrapper_running_check_timeout is not None:
            self.wrapper_running_check_timeout = wrapper_running_check_timeout
        if wrapper_finish_check_fn is not None:
            self.wrapper_finish_check_fn = wrapper_finish_check_fn
        if wrapper_finish_check_timeout is not None:
            self.wrapper_finish_check_timeout = wrapper_finish_check_timeout
        if wrapper_check_poll_interval is not None:
            self.wrapper_check_poll_interval = wrapper_check_poll_interval

        # Now save the perturbation into a file for the simulation to read in
        if u_pert is not None:
            if u_pert_fn is None:
                raise ValueError("The perturbation file name is not specified!")
            if pathlib.Path(self.base_dir + "/" + u_pert_fn).exists():
                # warnings.warn("The perturbation file already exists! Overwriting it.")
                os.remove(self.base_dir + "/" + u_pert_fn)
            with h5py.File(self.base_dir + "/" + u_pert_fn, 'w') as f:
                f.create_dataset('u_pert', data=u_pert)
            # Check whether the input parameter includes the perturbation file name
            if u_pert_fn not in params.values():
                raise ValueError("The perturbation file name is not included in the input parameter!")
        else:
            if u_pert_fn is not None:
                raise ValueError("The perturbation file name is specified, but the perturbation is not!")
            if self.ut1_unperturbed_fn is not None and self.t1 == t1:
                # if the unperturbed solution at t1 is already computed, then no need to rerun sim, jut return it
                t1_infile = self.yt_read_parameter(self.base_dir, self.ut1_unperturbed_fn, 'current_time')
                if np.isclose(t1, t1_infile, atol=np.min((t1, t1_infile)) * 1e-2):  # allow 1% tolerance due to timestep
                    ut = self.yt_read_solution(self.base_dir, self.ut1_unperturbed_fn, self.grow_var,
                                               self.yt_derived_fields)
                    logging.debug("The unperturbed solution at t1 is already computed, and "
                                  "is read from file {}.".format(self.ut1_unperturbed_fn))
                    return ut
                else:
                    warnings.warn("The unperturbed solution at t1 = %f is already computed, but the time does not "
                                  "match the requested t1 = %f!  Deleting it now for safety and will recompute it." %
                                  (t1_infile, t1))
                    os.remove(self.base_dir + "/" + self.ut1_unperturbed_fn)
                    self.t1 = None
                    self.ut1_unperturbed_fn = None

        # Now update the parameter file
        update_parameter(self.base_dir + "/" + self.param_fn, params)
        if pathlib.Path(self.base_dir + "/" + ut_fn).exists():
            # warnings.warn("The evolving state file already exists! Deleting it now for safety.")
            os.remove(self.base_dir + "/" + ut_fn)

        # Now start the simulation
        self.run_simulation_with_python_wrapper()

        if not pathlib.Path(self.base_dir + "/" + ut_fn).exists():
            raise ValueError("The evolving state file might be generated, but you might guess the file name wrong!")
        logging.debug("The ut state is evolved from the basic state u0 and saved as {}.".format(ut_fn))

        # Delete the perturbation file
        if u_pert_fn is not None:
            os.remove(self.base_dir + "/" + u_pert_fn)

        # Delete the file specified by delete_fn for each run, in case some log file grows too large
        if delete_fn is not None:
            # tell delete_fn is a string or list
            if isinstance(delete_fn, str):
                delete_fn = [delete_fn]
            for fn in delete_fn:
                os.remove(self.base_dir + "/" + fn)

        if u_pert is None and self.ut1_unperturbed_fn is None:
            # if the unperturbed solution at t1 is not saved, then save the current state as the unperturbed solution
            self.ut1_unperturbed_fn = ut_fn + "_unperturbed"
            self.t1 = t1
            os.system("cp " + self.base_dir + "/" + ut_fn + " " + self.base_dir + "/" + self.ut1_unperturbed_fn)

        # Return the evolving state ut
        ut = self.yt_read_solution(self.base_dir, ut_fn, self.grow_var, self.yt_derived_fields)

        # Change back to old base_dir and delete fork_id
        if fork_id is not None:
            shutil.rmtree(self.base_dir,  ignore_errors=True)
            self.base_dir = old_base_dir
        return ut

    @staticmethod
    def yt_read_solution(base_dir, fn, grow_var, derived_fields=None):
        # Return the evolving state ut as one-dimensional array, since its spatial info is not needed
        ds = yt.load(base_dir + "/" + fn)
        if derived_fields is not None:
            derived_fields(ds)
        solution = ds.all_data()[grow_var].v
        del ds
        return solution

    @staticmethod
    def yt_read_parameter(base_dir, fn, param_name):
        ds = yt.load(base_dir + "/" + fn)
        if param_name == "current_time":
            param = ds.current_time.v
        else:
            param = ds.parameters[param_name]
        del ds
        return param

    def make_fork_dir(self, fork_dir: str):
        # copy all the files in the base_dir to the fork_id, but excluding folders
        if pathlib.Path(fork_dir).exists():
            shutil.rmtree(fork_dir, ignore_errors=True)
        os.mkdir(fork_dir)
        # subprocess is needed to enter the fork_dir and create symbolic links
        for fn in self.link_list:
            subprocess.run(["ln", "-s", "../" + fn, "."], cwd=fork_dir)
        os.system("cp -r " + self.base_dir + "/" + (" " + self.base_dir + "/").join(self.copy_list) + " " + fork_dir)
        return

    # def remove_fork_dirs(self):
    #     # remove all fork directories
    #     for fn in os.listdir(self.base_dir):
    #         if fn.startswith("fork_"):
    #             shutil.rmtree(self.base_dir + "/" + fn, ignore_errors=True)
    #     return
