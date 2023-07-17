import numpy as np
import matplotlib.pyplot as plt
from solvers import evolve

time = 6.
delta_t = 0.1
nt = int(time / delta_t + 1)
L = 100
delta_x = 1.0
vis = 0.5
x = np.arange(0, L + 1) * delta_x
u0 = np.sin(2 * np.pi * x / L)
fx = evolve(u0, nt, vis, delta_t, delta_x)
plt.plot(x, u0)
plt.plot(x, fx)
plt.show()
