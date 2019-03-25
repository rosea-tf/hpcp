"""
Plotting shear wave decay

@author: AlexR
"""

#%%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import _pickle as pickle
import numpy as np
import os
import gzip

#%% LOAD DATA

res = pickle.load(gzip.open(os.path.join('pickles', 'shearwave.pkl.gz'), 'rb'))

u_init = res['u_initial']
lat_x, lat_y = u_init.shape[0:2]
epsilon = res['epsilon']
t_hist = res['t_hist']
omegas = res['omegas']
amplitude_hists = res['amplitude_hists']
k = res['k']

#find reference position for sin wave peak
sin_peak_pos_y = np.argmax(u_init[0, :, 0])  #r0

#%% INITIAL CONDITIONS

plt.rcParams.update(plt.rcParamsDefault)

plt.clf()
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

ax[0].set_title('Initial shear wave magnitude')
ax[0].set_xlabel('$u_x$')
ax[0].set_ylabel('$y$')
ax[0].set_yticks([])
ax[0].plot(res['u_initial'][0, :, 0], np.arange(lat_y))

ax[1].set_title('Initial streaming pattern in lattice')
ax[1].set_xlabel('$x$')
ax[1].set_xticks([])
ax[1].streamplot(
    np.arange(lat_x),
    np.arange(lat_y),
    *np.transpose(u_init, [2, 1, 0]),
    linewidth=(3 / epsilon) * np.linalg.norm(u_init, axis=2).T)

plt.savefig('./plots/shearwave_initial.png', dpi=150, bbox_inches='tight')

#%% ANALYTICAL PREDICTIONS
plt.clf()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].set_title('Amplitude of shear wave over time')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$u_x(t) / u_x(0)$')

ax[1].set_title('Kinematic Viscosity')
ax[1].set_xlabel('$\omega$')
ax[1].set_ylabel('$\\nu$')

viscosity_obs = []
viscosity_calcs = []

for omega in omegas:

    amplitude_hist = amplitude_hists[omega]

    amp_peak_ppn_hist = amplitude_hist[:, sin_peak_pos_y] / amplitude_hist[
        0, sin_peak_pos_y]

    # theoretical viscosity
    viscosity_calc = (1 / 3) * ((1 / omega) - (1 / 2))

    viscosity_calcs.append(viscosity_calc)

    viscosity_obs.append(
        -(1 / ((k**2) * (t_hist[-1] + 1))) * np.log(amp_peak_ppn_hist[-1]))

    # theoretical sine wive amplitude
    alpha = viscosity_calc * (k**2)
    amp_peak_ppn_calc = np.exp(-alpha * t_hist)

    ax[0].scatter(
        t_hist,
        amp_peak_ppn_hist,
        label='Observed ($\omega={}$)'.format(omega),
        alpha=0.1)

    ax[0].plot(
        t_hist,
        amp_peak_ppn_calc,
        # linestyle='-.',
        label='Predicted ($\omega={}$)'.format(omega))

ax[0].legend()

ax[1].scatter(omegas, viscosity_obs, label='Observed', c='orange')
omega_r = np.linspace(0.25, 1.75)
viscosity_calc_r = (1 / 3) * ((1 / omega_r) - (1 / 2))

ax[1].plot(omega_r, viscosity_calc_r, label='Predicted')
ax[1].legend()

plt.savefig('./plots/shearwave_predictions.png', dpi=150, bbox_inches='tight')

#%% SHEARWAVE 3D
plt.clf()
fig = plt.figure(figsize=(10, 4))

# Make data.
y = np.arange(lat_y)
yy, tt = np.meshgrid(y, t_hist)

# set up the axes for the first plot
for i, omega in enumerate([0.25, 1.0]):
    ax = fig.add_subplot(1, 2, 1 + i, projection='3d')

    ax.plot_surface(
        yy,
        tt,
        amplitude_hists[omega] / epsilon,
        cmap=cm.get_cmap('coolwarm'),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True)
    ax.set_title('$\omega={}$'.format(omega))
    ax.set_xlabel('$y$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('Amplitude')

plt.tight_layout()
plt.savefig('./plots/shearwave_time.png', dpi=150, bbox_inches='tight')

#%%
print("Plotting complete. Results saved in ./plots/")