import numpy as np
import matplotlib.pyplot as plt
from cnop_methods import spg2_defn
from utils import usphere_sample
from solvers.flash import Flash

t0 = 10.
flash = Flash(t0, "../flash4/object", "mpirun -np 4 ./flash4", "cnop1d", "dens")
u_pert = flash.generate_u_pert(pert_mag=0.1)
t1 = 30.
u_pert_best, j_best = spg2_defn(flash, u_pert, t1)
# save the result
np.savez("flash_u_pert_best.npz", u_pert_best=u_pert_best, j_best=j_best)