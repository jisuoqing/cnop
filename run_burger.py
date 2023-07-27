import numpy as np
import matplotlib.pyplot as plt
from cnop_methods import spg2_defn
from utils import usphere_sample
from solvers.burgers import Burgers

# parameters
t1 = 6.
delta_t = 0.1
nt = int(t1 / delta_t + 1)
L = 100
delta_x = 1.0
vis = 0.5
x = np.arange(0, L + 1) * delta_x
u_init = np.sin(2 * np.pi * x / L)
u_pert = np.zeros(u_init.shape)
u_pert[1:-1] = usphere_sample(len(u_init)-2)


# find initial conditions---u0_basic----以模式第58时刻的状态为初始条件 然后在此基础上添加扰动 再求CNOP
nt0 = 58
t0 = nt0 * (t1 / nt)

process = Burgers(u_init, t0, vis=vis, delta_t=delta_t, delta_x=delta_x)
u0_best, j_best = spg2_defn(process, u_pert, t1)

plt.plot(x, u0_best)
plt.show()

