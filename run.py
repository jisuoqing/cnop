import numpy as np
import matplotlib.pyplot as plt
from cnop_methods import spg2_defn
from burgers_lib import solve_burgers
from utils import usphere_sample

# parameters
time = 6.
delta_t = 0.1
nt = int(time / delta_t + 1)
L = 100
delta_x = 1.0
vis = 0.5
x = np.arange(0, L + 1) * delta_x
u0 = np.sin(2 * np.pi * x / L)
u_pert = np.zeros(u0.shape)
u_pert[1:-1] = usphere_sample(len(u0)-2)

# find initial conditions---u0_basic----以模式第58时刻的状态为初始条件 然后在此基础上添加扰动 再求CNOP
T1 = 58
u0_basic = solve_burgers(u0, T1, vis, delta_t, delta_x)

u0_best, j_best = spg2_defn(u0_basic, u_pert, nt, vis, delta_t, delta_x)

ut0 = solve_burgers(u0_basic, nt, vis, delta_t, delta_x)
ut_best = solve_burgers(u0_basic + u0_best, nt, vis, delta_t, delta_x)

plt.plot(x, ut0)
plt.plot(x, ut_best)
plt.show()

