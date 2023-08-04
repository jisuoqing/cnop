import numpy as np
from solvers.flash import Flash
from cnop_methods import Spg2Defn

t0 = 10.
flash = Flash(t0, "../flash4/object", "mpirun -np 4 ./flash4", "cnop1d", "dens", "dens")
u_pert = flash.generate_u_pert(pert_mag=0.1)
t1 = 30.
spg2 = Spg2Defn(flash, u_pert, t1)
# save the result
np.savez("flash_u_pert_best.npz", u_pert_best=spg2.u_pert_best, j_best=spg2.j_best)