from solvers.simulation import Simulation


class Flash(Simulation):
    def __init__(self, t0, base_dir, exec_cmd, basename, cnop_var):
        init_params = {
            "restart": ".false.",
            "checkpointFileNumber": 0,
            "plotFileNumber": 0,
            "cnop_var": cnop_var,
            "tmax": t0,
            # Do not do inject perturbation when evolving to basic state, for safety
            "cnop_doInject": ".true.",
            # Do not produce any output files except the last one
            "checkpointFileIntervalStep": 0,
            "checkpointFileIntervalTime": 0.,
            "plotFileIntervalStep": 0,
            "plotFileIntervalTime": 0.,
        }
        self.basename = basename
        u0_fn = "%s_hdf5_chk_0001" % self.basename
        super().__init__(None, base_dir, exec_cmd, init_params, "flash.par", u0_fn, cnop_var)
        return

    def proceed(self, t1, u_pert=None):
        exec_cmd = self.exec_cmd  # Flash does not need a different command for restarting
        if u_pert is None:
            cnop_do_inject = ".false."
            u_pert_fn = None
        else:
            cnop_do_inject = ".true."
            u_pert_fn = "u_pert.h5"
        params = {
            "restart": ".true.",
            "checkpointFileNumber": 1,  # restart from _chk_0001
            "plotFileNumber": 0,
            "cnop_var": self.cnop_var,
            "tmax": t1,
            "cnop_doInject": cnop_do_inject,
            "cnop_injectFile": u_pert_fn,
            # Do not produce any output files except the last one
            "checkpointFileIntervalStep": 0,
            "checkpointFileIntervalTime": 0.,
            "plotFileIntervalStep": 0,
            "plotFileIntervalTime": 0.,
        }
        # since we restart from "_chk_0001", new file should be named as "_chk_0002"
        ut_fn = "%s_hdf5_chk_0002" % self.basename  # based on FLash dataset naming format, say "cnop1d_hdf5_chk_0000"
        ut = super().proceed_simulation(params, exec_cmd, u_pert, u_pert_fn, ut_fn)
        return ut
