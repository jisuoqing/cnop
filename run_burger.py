import numpy as np
import matplotlib.pyplot as plt
from utils import usphere_sample
from solvers.burgers import Burgers
from cnop_methods import Spg2Defn

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
spg2 = Spg2Defn(process, u_pert, t1)


plt.subplot(211)
plt.plot(x, process.u0, label=r"$u_0 (t_0)$")
plt.plot(x, process.ut1_unperturbed, label=r"$u(u_0, t_0 + \Delta t)$")
plt.plot(x, process.proceed(t1, u_pert=spg2.u_pert_best), label=r"$u(u_0 + \delta u_{\rm best}, t_0 + \Delta t)$")
plt.legend()
plt.xlabel("t")
plt.subplot(212)
plt.plot(x, spg2.u_pert_best, label=r"$\delta u_{\rm best}$")
plt.legend()
plt.xlabel("t")
plt.tight_layout()
plt.savefig("burgers.png", dpi=300, bbox_inches="tight")

