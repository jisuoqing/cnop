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

# print(u0_best, j_best)

# plot the figure
# fig = plt.figure('units','normalized','position',np.array([0.0,0.0,0.6,0.8]))
# plt.plot(x,u0_best,'-o')
# plt.grid('on')
# plt.axis(np.array([1,101,- 0.00025,0.00025]))
# set(gca,'xtick',(np.arange(1,101+20,20)),'xticklabel',(np.arange(0,100+20,20)))
# set(gca,'ytick',(np.arange(- 0.00025,0.00025+0.000125,0.000125)),'yticklabel',(np.array(['-2.5\times10^{-4}','-1.25\times10^{-4}','0','1.25\times10^{-4}','2.5\times10^{-4}'])))
# set(gca,'fontsize',15)
# plt.xlabel('$x$ $(m)$','Interpreter','latex','fontsize',35)
# plt.ylabel('CNOP  $u_0$ ($m/s$)','Interpreter','latex','fontsize',35)
# plt.title('Definition','Interpreter','latex','fontsize',40)
# #set(gca,'fontsize',12);
#
# # draw the norm of perturbation
# U = burgers(U0_basic,T,vis,delta_t,delta_x)
# U_best = burgers(U0_basic + u0_best,T,vis,delta_t,delta_x)
# u_prime = U_best - U
# u_defn_norm = np.zeros((T,1))
# for i in np.arange(1,T+1).reshape(-1):
#     sum = 0
#     for j in np.arange(1,d+1).reshape(-1):
#         sum = sum + u_prime(i,j) ** 2
#     u_defn_norm[i,1] = sum

#save('u_defn.mat','u_defn_norm','-v7.3','-nocompression')
