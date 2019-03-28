#%% IMPORTS

import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import _pickle as pickle
from utils import plot_save

#%% LOAD DATA

res = pickle.load(
    gzip.open(os.path.join('pickles', 'sinedensity.pkl.gz'), 'rb'))

x = np.arange(res['lat_x'])
t_hist = res['t_hist']
MAX_SP = 400
t_hist_sp = t_hist[:MAX_SP]
xx, tt = np.meshgrid(x, t_hist_sp)

# these will hold the summaries for the 2D plots
density_peak_hist = {}
velocity_peak_hist = {}

plt.rcParams.update(plt.rcParamsDefault)

# we plot only for this value, though others are available in results set
omega = 0.5

density_hist = res['density_hists'][omega]
velocity_hist = res['velocity_hists'][omega]

#%% PLOT INITIAL CONDITIONS
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Initial density $\\rho$ (magnitude)')
ax.set_xlabel('$x$')
ax.set_ylabel('$u_x$')
ax.set_xticks([])
ax.plot(x, density_hist[0])

ax = fig.add_subplot(1, 2, 2)
ax.set_title('Initial density $\\rho$ (lattice)')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(np.broadcast_to(density_hist[0, :, None], shape=(400, 300)).T)
fig.colorbar(im)

plot_save(fig, 'sinedensity_initial.png')

#%% 3D FIGURE
fig = plt.figure(figsize=(10, 4))

# density plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.set_title('Density $\\rho$ over time')
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$\\rho$')

surf = ax.plot_surface(
    xx,
    tt,
    density_hist[:MAX_SP],
    rstride=1,
    cstride=1,
    cmap=cm.get_cmap('viridis'),
    linewidth=0,
    antialiased=True)

# velocity plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.set_title('Velocity $\mathbf{u}_x$ over time')
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$\mathbf{u}_x$')

ax.plot_surface(
    xx,
    tt,
    velocity_hist[:MAX_SP],
    rstride=1,
    cstride=1,
    cmap=cm.get_cmap('coolwarm'),
    linewidth=0,
    antialiased=True)

plot_save(fig, 'sinedensity_3d.png')

#%% 2D PLOT
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# density plot
ax[0].set_title('$Var(\\rho)$ over time')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$\\rho(x_{ref})$')

for omega in res['omegas']:
    density_hist = res['density_hists'][omega]
    density_init_argmax = np.argmax(density_hist[0])
    ax[0].plot(
        t_hist, density_hist.std(axis=1)**2, label='$\omega={}$'.format(omega))
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# velocity plot
ax[1].set_title('$Var(\mathbf{u}_x)$ over time')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$\mathbf{u}_x(x_{ref})$')

for omega in res['omegas']:
    velocity_hist = res['velocity_hists'][omega]
    ax[1].plot(
        t_hist,
        velocity_hist.std(axis=1)**2,
        label='$\omega={}$'.format(omega))
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax[1].legend()

plot_save(fig, 'sinedensity_2d.png')
