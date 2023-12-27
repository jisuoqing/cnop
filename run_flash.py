from solvers.flash import Flash
from cnop_methods import Spg2Defn
import numpy as np
import logging

# logging.basicConfig(level=logging.DEBUG)


def derived_fields(ds):
    def _cool_dens(field, data):
        return data["dens"] * (data["temp"] <= 2)

    ds.add_field(("gas", "cool_dens"), function=_cool_dens, units="code_mass/code_length**3",
                 force_override=True, sampling_type="cell")


# avoid running the following script when importing it by multiprocessing
if __name__ == "__main__":

    problem = "cnop1d"

    if problem == "cnop1d":

        t0 = 10.
        flash = Flash(t0, "../flash4/object", "./flash4", 2,
                      "cnop1d", "dens", "dens",
                      # shorter polling interval since the simulation is fast
                      wrapper_check_poll_interval=0.1
                      )
        u_pert = flash.generate_u_pert(pert_delta=0.1)
        t1 = 30.
        spg2 = Spg2Defn(flash, u_pert, t1, 8e-6)
        # save the result
        if flash.mpi_rank == 0:
            np.savez("flash_u_pert_best.npz", u_pert_best=spg2.u_pert_best, j_best=spg2.j_best)

    elif problem == "cloud_crushing":

        t0 = 0.0
        flash = Flash(t0, "../flash4/object", "./flash4", 1,
                      "cloud_crushing", "dens", "cool_dens",
                      wrapper_check_poll_interval=10,
                      wrapper_finish_check_timeout=np.inf,
                      yt_derived_fields=derived_fields,
                      link_list=["cool_func.dat"])
        dens = flash.get_covering_grid("density")
        pert_mask = (dens > dens.max() * 0.1)
        u_pert = flash.generate_u_pert(pert_delta=dens.max() * 0.1, pert_mask=pert_mask)
        t1 = 50.
        # first, feeding in all-space perturbations to make sure the sim likes it
        # flash.proceed(t1 * 0.1, u_pert=u_pert, fork_id=100)

        spg2 = Spg2Defn(flash, u_pert, t1, pert_delta=dens.max() * 0.1, pert_mask=pert_mask, grad_epsilon=1.e-4)
        # save the result
        if flash.mpi_rank == 0:
            np.savez("flash_u_pert_best.npz", u_pert_best=spg2.u_pert_best, j_best=spg2.j_best)
