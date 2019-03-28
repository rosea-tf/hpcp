#%% IMPORTS

import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import _pickle as pickle
from utils import plot_save

# %% SETUP

res = pickle.load(
    gzip.open(os.path.join('pickles', 'poiseuille.pkl.gz'), 'rb'))

t_hist_hf = res['t_hist_hf']
t_hist_sp = res['t_hist_sp']
walls = res['walls']

halfway_vel_hist = res['halfway_vel_hist']
flow_hist = res['flow_hist']
lat_x = res['lat_x']
lat_y = res['lat_y']
inflow = res['inflow']
x = np.arange(lat_x)
y = np.arange(lat_y)

yy, tt = np.meshgrid(y, t_hist_hf)

plt.rcParams.update(plt.rcParamsDefault)

#%% EVOLUTION OF FLOW OVER TIME, AND PARABOLA
plt.clf()
fig, ax = plt.subplots(1, 2, sharey=True, figsize=[10, 4])

ax[0].set_title('$u_x$ velocity over time')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$u_x$')
ax[0].plot(
    t_hist_hf, halfway_vel_hist[:, lat_y // 2, 0], label='Center of $y$-axis')
ax[0].plot(
    t_hist_hf,
    np.mean(halfway_vel_hist[:, :, 0], axis=1),
    label='Average over $y$-axis')
ax[0].legend()

ax[1].set_title('Final $u_x$ velocity')
ax[1].set_xlabel('$y$')
ax[1].plot(
    y, halfway_vel_hist[-1, :, 0], label='Observed', alpha=0.5, linewidth=5)

throughflow_halfway_avg = halfway_vel_hist[-1, :, 0].mean()

para_calc = (6 * throughflow_halfway_avg / ((lat_y - 1)**2)) * y * (
    (lat_y - 1) - y)

ax[1].plot(para_calc, label='Calculated')
ax[1].legend()

plot_save(fig, 'poiseuille_half2d.png')

#%% 3D REPRESENTATION
plt.clf()
fig = plt.figure()

# set up the axes for the first plot
ax = fig.add_subplot(1, 1, 1, projection='3d')

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

plot_save(fig, 'poiseuille_half3d.png')

#%% STREAMPLOT
plt.clf()
fig, axc = plt.subplots(3, 3, sharex=True, sharey=True, figsize=[10, 8])

for a in axc[-1]:
    a.set_xlabel('$x$')
for a in axc[:, 0]:
    a.set_ylabel('$y$')

ax = axc.reshape(-1)
for i in range(9):
    ax[i].set_title('t={}'.format(t_hist_sp[i]))
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    ax[i].scatter(*walls.T, marker='s', s=1, color='red')

    ax[i].streamplot(
        x,
        y,
        *np.transpose(flow_hist[i], [2, 1, 0]),
        linewidth=(100) * np.linalg.norm(flow_hist[i], axis=2).T)

plot_save(fig, 'poiseuille_streamtime.png')
