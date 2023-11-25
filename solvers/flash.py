from solvers.simulation import Simulation


class Flash(Simulation):
    def __init__(self, t0, base_dir, exec_command, wrapper_nproc,
                 basename, pert_var, grow_var, exec_finish_check_poll_interval=1.,
                 yt_derived_fields=None,
                 link_list=None, copy_list=None):
        # TODO: add a warning of wrapper_nproc != iprocs * jprocs * kprocs
        init_params = {
            "restart": ".false.",
            "checkpointFileNumber": 0,
            "plotFileNumber": 0,
            "cnop_pert_var": pert_var,
            # "cnop_grow_var" no need to include grow_var in sim params since it is handled by Python post-processing
            "tmax": t0,
            # Do not do inject perturbation when evolving to basic state u0
            # Although injection is not performed when restart = false, we still turn off cnop_doInject for safety
            "cnop_doInject": ".false.",
            # Do not produce any output files except the last one
            "checkpointFileIntervalStep": 0,
            "checkpointFileIntervalTime": 0.,
            "plotFileIntervalStep": 0,
            "plotFileIntervalTime": 0.,
        }
        self.basename = basename
        u0_fn = "%s_hdf5_chk_0001" % self.basename
        if copy_list is None:
            copy_list = []
        copy_list.append("flash4")  # Flash executable

        exec_args = ""
        wrapper_finish_check_fn = u0_fn
        if exec_finish_check_poll_interval is None:
            exec_finish_check_poll_interval = 1.
        super().__init__(None, base_dir,
                         exec_command, exec_args, wrapper_nproc,
                         wrapper_finish_check_fn, exec_finish_check_poll_interval,
                         init_params, "flash.par", u0_fn,
                         pert_var, grow_var, yt_derived_fields=yt_derived_fields,
                         link_list=link_list, copy_list=copy_list)
        return

    def proceed(self, t1, u_pert=None, u_pert_fn="u_pert.h5", fork_id=None):
        exec_command = self.exec_command  # Flash does not need a different command for restarting
        wrapper_args = self.wrapper_args
        wrapper_nproc = self.wrapper_nproc
        if u_pert is None:
            cnop_do_inject = ".false."
            u_pert_fn = None
        else:
            cnop_do_inject = ".true."
        params = {
            "restart": ".true.",
            "checkpointFileNumber": 1,  # restart from _chk_0001
            "plotFileNumber": 0,
            "cnop_pert_var": self.pert_var,
            # "cnop_grow_var" no need to include grow_var in sim params since it is handled by Python post-processing
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

        # Delete FLASH log and .dat files which will not be overwritten and will increase in size if not deleted
        delete_fn = ["flash.dat", self.basename + ".log"]
        wrapper_finish_check_fn = ut_fn

        ut = super().proceed_simulation(params, t1, exec_command, wrapper_args, wrapper_nproc, wrapper_finish_check_fn,
                                        self.exec_finish_check_poll_interval,
                                        u_pert, u_pert_fn, ut_fn, delete_fn, fork_id=fork_id)
        return ut
