import numpy as np
from burgers_lib import solve_burgers
import matplotlib.pyplot as plt

time = 6.
delta_t = 0.1
nt = int(time / delta_t + 1)
L = 100
delta_x = 1.0
vis = 0.5
x = np.arange(0, L + 1) * delta_x
U0 = np.sin(2 * np.pi * x / L)
fx = solve_burgers(U0, nt, vis, delta_t, delta_x)
plt.plot(x, U0)
plt.plot(x, fx)
plt.show()
