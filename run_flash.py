import numpy as np
from solvers.flash import Flash
from cnop_methods import Spg2Defn

problem = "cloudcrushing"

if problem == "cnop1d":

    t0 = 10.
    flash = Flash(t0, "../flash4/object", "mpirun -np 4 ./flash4",
                  "cnop1d", "dens", "dens")
    u_pert = flash.generate_u_pert(pert_mag=0.1)
    t1 = 30.
    spg2 = Spg2Defn(flash, u_pert, t1)
    # save the result
    np.savez("flash_u_pert_best.npz", u_pert_best=spg2.u_pert_best, j_best=spg2.j_best)

elif problem == "cloudcrushing":
    def derived_fields():
        import yt

        @yt.derived_field(name="cool_dens", units="code_mass/code_length**3", force_override=True, sampling_type="cell")
        def _cool_dens(field, data):
            return data["dens"] * (data["temp"] <= 2)


    print("Initializing Flash")
    t0 = 0.01
    flash = Flash(t0, "../flash4/object", "mpirun --oversubscribe -np 48 ./flash4",
                  "cloud_crushing", "dens", "cool_dens", derived_fields=derived_fields)
    print("Generating perturbation")
    u_pert = flash.generate_u_pert(pert_mag=0.1)
    t1 = 0.1
    spg2 = Spg2Defn(flash, u_pert, t1)
    # save the result
    np.savez("flash_u_pert_best.npz", u_pert_best=spg2.u_pert_best, j_best=spg2.j_best)
