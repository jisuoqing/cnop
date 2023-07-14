import numpy as np
import burgers
import matplotlib.pyplot as plt

time = 6.
delta_t = 0.1
nt = int(time / delta_t + 1)
L = 100
delta_x = 1.0
vis = 0.5
x = np.arange(0, L + 1) * delta_x
U0 = np.sin(2 * np.pi * x / L)
fx = burgers.solve_burgers(nt, U0, vis, delta_t, delta_x)
plt.plot(x, U0)
plt.plot(x, fx)
plt.show()
