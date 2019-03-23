#%% CHARTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import _pickle as pickle
import os
import gzip

res = pickle.load(
    gzip.open(os.path.join('pickles', 'sinedensity.pkl.gz'), 'rb'))

# make data
MAX_SP = 400

x = np.arange(res['lat_x'])
t_hist = res['t_hist']
t_hist_sp = t_hist[:MAX_SP]
xx, tt = np.meshgrid(x, t_hist_sp)


# these will hold the summaries for the 2D plots
density_peak_hist = {}
velocity_peak_hist = {}

plt.rcParams.update(plt.rcParamsDefault)

for omega in res['omegas']:
    density_hist = res['density_hists'][omega]
    velocity_hist = res['velocity_hists'][omega]

    #3D FIGURE
    fig = plt.figure(figsize=(10, 4))

    # DENSITY PLOT
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

    # VELOCITY PLOT
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

    plt.tight_layout()
    plt.savefig(
        './plots/sinedensity_3d_{}.png'.format(omega),
        dpi=150,
        bbox_inches='tight')

# 2D PLOT
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# DENSITY PLOT
ax[0].set_title('$Var(\\rho)$ over time')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$\\rho(x_{ref})$')

for omega in res['omegas']:
    density_hist = res['density_hists'][omega]
    density_init_argmax = np.argmax(density_hist[0])
    ax[0].plot(t_hist, density_hist.std(axis=1)**2, label='$\omega={}$'.format(omega))
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax[1].set_title('$Var(\mathbf{u}_x)$ over time')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$\mathbf{u}_x(x_{ref})$')

for omega in res['omegas']:
    velocity_hist = res['velocity_hists'][omega]
    ax[1].plot(t_hist, velocity_hist.std(axis=1)**2, label='$\omega={}$'.format(omega))
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax[1].legend()

plt.tight_layout()
plt.savefig('./plots/sinedensity_2d.png', dpi=150, bbox_inches='tight')

print("Plotting complete. Results saved in ./plots/")


#%%
