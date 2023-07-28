from sim_controller import update_parameter
import subprocess
import os
import warnings
import shlex


class Simulation:
    def __init__(self, u_init_fn, t0, base_dir, exec_cmd, init_params, param_fn, u0_file):
        # u_init_fn is usually empty since the IC is usually generated by the solver
        # However, sometimes the IC is generated by the user (such as Gizmo), in which case u_init_fn is not empty
        if u_init_fn is not None:
            if not os.path.exists(base_dir + "/" + u_init_fn):
                raise ValueError("The initial condition file does not exist!")
            self.u_init_fn = u_init_fn
        self.t0 = t0
        self.base_dir = base_dir
        self.exec_cmd = exec_cmd
        self.init_params = init_params
        self.param_fn = param_fn
        # u0_file is the file name of the basic state, which must match the output file name of the solver
        self.u0_file = u0_file
        if os.path.exists(self.base_dir + "/" + self.u0_file):
            warnings.warn("The basic state file already exists! Overwriting it.")
        # Now evolve the initial condition to t0, to obtain u0 which is the basic state
        update_parameter(self.base_dir + "/" + self.param_fn, self.init_params)
        with open('%s_basic_state_stdout.txt' % self.__class__.__name__, 'w') as stdout_file, \
                open('%s_basic_state_stderr.txt' % self.__class__.__name__, 'w') as stderr_file:
            exec_args = shlex.split(self.exec_cmd)
            process = subprocess.Popen(exec_args, stdout=stdout_file, stderr=stderr_file, shell=False,
                                       cwd=self.base_dir)
            process.wait()
        if process.returncode != 0:
            raise ValueError("The solver is not working properly, and no basic state file is generated!")
        elif not os.path.exists(self.base_dir + "/" + self.u0_file):
            raise ValueError("The basic state file might be generated, but you might guess the file name wrong!")
        else:
            print("The basic state is evolved from the initial condition to time {}.".format(self.t0))

    def proceed(self, t1, exec_cmd=None, u_pert=None):
        if exec_cmd is not None:
            # a different command is needed for restarting
            self.exec_cmd = exec_cmd
        if u_pert is None:
            update_parameter(self.base_dir + "/" + self.param_fn, {"restart": ".true."})
        else:
            update_parameter(self.base_dir + "/" + self.param_fn, {"restart": ".true."})
            update_parameter(self.base_dir + "/" + self.param_fn, {"perturbation": u_pert})
        with open('%s_evolving_state_stdout.txt' % self.__class__.__name__, 'w') as stdout_file, \
                open('%s_evolving_state_stderr.txt' % self.__class__.__name__, 'w') as stderr_file:
            exec_args = shlex.split(self.exec_cmd)
            process = subprocess.Popen(exec_args, stdout=stdout_file, stderr=stderr_file, shell=False,
                                       cwd=self.base_dir)
            process.wait()
        if process.returncode != 0:
            raise ValueError("The solver is not working properly, and no evolving state file is generated!")
        return
