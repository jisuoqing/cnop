from sim_controller import update_parameter, get_system_info, find_latest_checkpoint, load_checkpoint
import subprocess
import os
import warnings
import shlex
import yt
import numpy as np
import h5py
import pathlib
import shutil

yt.set_log_level("error")


class Simulation:
    def __init__(self, u_init_fn: str, base_dir: str, exec_cmd: str, init_params: dict, param_fn: str, u0_fn: str,
                 pert_var: str, grow_var: str, yt_derived_fields: callable = None,
                 link_list: list = None, copy_list: list = None):
        """
        :param u_init_fn: Initial condition file name. If None, then the initial condition is generated by the solver
            (e.g., Flash, Athena). Otherwise, the initial condition is read from the file (e.g., Gizmo)
        :param base_dir: The directory where the simulation is run
        :param exec_cmd: The command to run the simulation
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
            print("The class is initialized with the checkpoint file {}.".format(self.restart_checkpoint_fn))
            return
        else:
            if u_init_fn is not None:
                if not pathlib.Path(base_dir + "/" + u_init_fn).exists():
                    raise ValueError("The initial condition file does not exist!")
                self.u_init_fn = u_init_fn
            self.base_dir = base_dir
            self.exec_cmd = exec_cmd
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

            if pathlib.Path(self.base_dir + "/" + self.u0_fn).exists():
                # warnings.warn("The basic state file already exists! Deleting it now for safety.")
                os.remove(self.base_dir + "/" + self.u0_fn)
            # Now evolve the initial condition to t0, to obtain u0 which is the basic state
            update_parameter(self.base_dir + "/" + self.param_fn, init_params)
            with open('%s_basic_state_stdout.txt' % self.__class__.__name__, 'w') as stdout_file, \
                    open('%s_basic_state_stderr.txt' % self.__class__.__name__, 'w') as stderr_file:
                exec_args = shlex.split(self.exec_cmd)
                process = subprocess.Popen(exec_args, stdout=stdout_file, stderr=stderr_file, shell=False,
                                           cwd=self.base_dir)
                process.wait()
            if process.returncode != 0:
                raise ValueError("The solver is not working properly, and no basic state file is generated!")
            elif not pathlib.Path(self.base_dir + "/" + self.u0_fn).exists():
                raise ValueError("The basic state file might be generated, but you might guess the file name wrong!")
            # print("The basic state u0 is evolved from the initial condition u_init and saved as {}.".format(
            # self.u0_fn))
            return

    def generate_u_pert(self, pert_mag):
        # generate a perturbation file with magnitude pert_mag
        # the perturbation file is saved as u_pert_fn
        ds = yt.load(self.base_dir + "/" + self.u0_fn)
        u_pert = np.random.uniform(-pert_mag, pert_mag, ds.domain_dimensions)
        del ds
        return u_pert

    def proceed_simulation(self, params, t1, exec_cmd=None, u_pert=None, u_pert_fn=None,
                           ut_fn=None, delete_fn=None, fork_id=None):
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

        if exec_cmd is not None:
            # if a different command is needed for restarting
            self.exec_cmd = exec_cmd
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
                    # print("The unperturbed solution at t1 is already computed, and is read from file {}.".format(
                    # self.ut1_unperturbed_fn))
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
        with open('%s_evolving_state_stdout.txt' % self.__class__.__name__, 'w') as stdout_file, \
                open('%s_evolving_state_stderr.txt' % self.__class__.__name__, 'w') as stderr_file:
            exec_args = shlex.split(self.exec_cmd)
            process = subprocess.Popen(exec_args, stdout=stdout_file, stderr=stderr_file, shell=False,
                                       cwd=self.base_dir)
            process.wait()

        if process.returncode != 0:
            print("The solver is not working properly! Dump system info and retrying...")
            get_system_info()
            print("Executing command: {}".format(self.exec_cmd))
            with open('%s_evolving_state_stdout.txt' % self.__class__.__name__, 'w') as stdout_file, \
                    open('%s_evolving_state_stderr.txt' % self.__class__.__name__, 'w') as stderr_file:
                exec_args = shlex.split(self.exec_cmd)
                process = subprocess.Popen(exec_args, stdout=stdout_file, stderr=stderr_file, shell=False,
                                           cwd=self.base_dir)
                process.wait()
            if process.returncode != 0:
                raise ValueError("The solver is not working properly after a 2nd trial, and no evolving state file is "
                                 "generated!")

        if not pathlib.Path(self.base_dir + "/" + ut_fn).exists():
            raise ValueError("The evolving state file might be generated, but you might guess the file name wrong!")
        # print("The ut state is evolved from the basic state u0 and saved as {}.".format(ut_fn))

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
            # shutil.rmtree(self.base_dir)
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
            shutil.rmtree(fork_dir)
        os.mkdir(fork_dir)
        # subprocess is needed to enter the fork_dir and create symbolic links
        for fn in self.link_list:
            subprocess.run(["ln", "-s", "../" + fn, "."], cwd=fork_dir)
        # p = subprocess.Popen(["ln", "-s", " ../" + " ../".join(self.link_list), '.'], cwd=fork_dir)
        # p.wait()
        os.system("cp -r " + self.base_dir + "/" + (" " + self.base_dir + "/").join(self.copy_list) + " " + fork_dir)
        return
