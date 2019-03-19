#%% CHARTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import argparse
import _pickle as pickle
import os

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "results1", type=str, help="filename of first pickled results set")
args = parser.parse_args()

# %%

args.results1 = 'poseuille.pkl'

# %%

res1 = pickle.load(open(os.path.join('pickles', args.results1), 'rb'))

halfway_vel_hist = res1['halfway_vel_hist']
walls = res1['walls']
flow_recordpoints = res1['flow_recordpoints']
flow_hist = res1['flow_hist']
# make data
x = np.arange(res1['lat_x'])
y = np.arange(res1['lat_y'])
t = res1['t_hist']
yy, tt = np.meshgrid(y, t)

plt.rcParams.update(plt.rcParamsDefault)

# %% Evolution of flow over time, and parabola
fig, ax = plt.subplots(1, 2, sharey=True, figsize=[10, 4])

ax[0].set_title('$u_x$ velocity over time')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$u_x$')
ax[0].plot(
    halfway_vel_hist[:, res1['lat_y'] // 2, 0], label='Center of $y$-axis')
ax[0].plot(
    np.mean(halfway_vel_hist[:, :, 0], axis=1), label='Average over $y$-axis')
ax[0].legend()

ax[1].set_title('Final $u_x$ velocity')
ax[1].set_xlabel('$x$')
ax[1].plot(halfway_vel_hist[-1, :, 0], label='Observed')
ax[1].legend()

# TODO: analytical solution!
# plt.plot(para_calc * (0.015 / 50))
# viscosity_calc = (1 / 3) * ((1 / omega) - (1 / 2))

# y_range = np.arange(lat_y)
# para_calc = (inflow / (2 * viscosity_calc)) * ((((lat_y / 2) ** 2)) - ((y_range - (lat_y / 2)) ** 2))

plt.savefig('./plots/poseuille_half2d.png', dpi=150, bbox_inches='tight')

#%%
fig, axc = plt.subplots(3, 3, sharex=True, sharey=True, figsize=[10, 8])

for a in axc[-1]:
    a.set_xlabel('$x$')
for a in axc[:, 0]:
    a.set_ylabel('$y$')

ax = axc.reshape(-1)
for i in range(9):
    ax[i].set_title('t={}'.format(flow_recordpoints[i]))
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    ax[i].scatter(*walls.T, marker='s', s=1, color='red')

    ax[i].streamplot(
        x,
        y,
        *np.transpose(flow_hist[i], [2, 1, 0]),
        linewidth=(100) * np.linalg.norm(flow_hist[i], axis=2).T)

plt.savefig('./plots/poseuille_streamtime.png', dpi=150, bbox_inches='tight')
#%%
fig = plt.figure(figsize=(10, 4))

# set up the axes for the first plot
ax = fig.add_subplot(1, 1, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
ax.plot_surface(
    yy,
    tt,
    halfway_vel_hist[:, :, 0],
    cmap=cm.get_cmap('Reds'),
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True)
ax.set_xlabel('$y$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u_x$')

plt.tight_layout()
plt.savefig('./plots/poseuille_half3d.png', dpi=150, bbox_inches='tight')


print("Plotting complete. Results saved in ./plots/")
