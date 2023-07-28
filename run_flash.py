import numpy as np
import matplotlib.pyplot as plt
from cnop_methods import spg2_defn
from utils import usphere_sample
from solvers.flash import Flash

t0 = 10.
process = Flash(t0, "../flash4/object", "mpirun -np 4 ./flash4", "cnop1d_hdf5_chk_0000", "dens")