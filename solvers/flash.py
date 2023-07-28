from solvers.simulation import Simulation


class Flash(Simulation):
    def __init__(self, t0, base_dir, exec_cmd, u0_file, cnop_var):
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
        super().__init__(None, t0, base_dir, exec_cmd, init_params, "flash.par", u0_file)
        return
