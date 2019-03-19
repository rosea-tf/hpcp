#%%
"""
Plotting shear wave decay

@author: AlexR
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import _pickle as pickle
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "results1",
    type=str,
    help="filename of first pickled results set",
    default='shearwave_1.pkl')
parser.add_argument(
    "results2",
    type=str,
    help="filename of second pickled results set",
    default='shearwave_2.pkl')
args = parser.parse_args()

res1 = pickle.load(open(os.path.join('pickles', args.results1), 'rb'))
res2 = pickle.load(open(os.path.join('pickles', args.results2), 'rb'))

# res1 = pickle.load(open('pickles/shearwave_1.pkl', 'rb'))
# res2 = pickle.load(open('pickles/shearwave_2.pkl', 'rb'))

u_init = res1['u_initial']
lat_x, lat_y = u_init.shape[0:2]
epsilon = res1['epsilon']
amp_peak_ppn_hist_1 = res1['amp_peak_ppn_hist']
amp_peak_ppn_hist_2 = res2['amp_peak_ppn_hist']
amp_peak_ppn_calc_1 = res1['amp_peak_ppn_calc']
amp_peak_ppn_calc_2 = res2['amp_peak_ppn_calc']
viscosity_hist_1 = res1['viscosity_hist']
viscosity_hist_2 = res2['viscosity_hist']
viscosity_calc_1 = res1['viscosity_calc']
viscosity_calc_2 = res2['viscosity_calc']
amplitude_hist_1 = res1['amplitude_hist']
amplitude_hist_2 = res2['amplitude_hist']
t_hist = res1['t_hist']
omega1 = res1['omega']
omega2 = res2['omega']

plt.rcParams.update(plt.rcParamsDefault)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

ax[0].set_title('Initial shear wave magnitude')
ax[0].set_xlabel('$u_x$')
ax[0].set_ylabel('$y$')
ax[0].set_yticks([])
ax[0].plot(res1['u_initial'][0, :, 0], np.arange(lat_y))

ax[1].set_title('Initial streaming pattern in lattice')
ax[1].set_xlabel('$x$')
ax[1].streamplot(
    np.arange(lat_x),
    np.arange(lat_y),
    *np.transpose(u_init, [2, 1, 0]),
    linewidth=(3 / epsilon) * np.linalg.norm(u_init, axis=2).T)

plt.savefig('./plots/shearwave_initial.png', dpi=150, bbox_inches='tight')

#%%

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].set_title('Amplitude of sine wave over time')
ax[0].plot(amp_peak_ppn_hist_1, label='Observed ($\omega={}$)'.format(omega1))
ax[0].plot(
    amp_peak_ppn_calc_1,
    linestyle='-.',
    label='Predicted ($\omega={}$)'.format(omega1))
ax[0].plot(amp_peak_ppn_hist_2, label='Observed ($\omega={}$)'.format(omega2))
ax[0].plot(
    amp_peak_ppn_calc_2,
    linestyle='-.',
    label='Predicted ($\omega={}$)'.format(omega2))
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('Proportion of initial amplitude')
# ax[0].legend()

ax[1].plot(viscosity_hist_1, label='Observed ($\omega={}$)'.format(omega1))
ax[1].axhline(
    viscosity_calc_1,
    linestyle='-.',
    color='orange',
    label='Predicted ($\omega={}$)'.format(omega1))
ax[1].plot(
    viscosity_hist_2,
    color='green',
    label='Observed ($\omega={}$)'.format(omega2))
ax[1].axhline(
    viscosity_calc_2,
    linestyle='-.',
    color='red',
    label='Predicted ($\omega={}$)'.format(omega2))
ax[1].set_title('Kinematic Viscosity')
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('Viscosity')
ax[1].legend()

plt.savefig('./plots/shearwave_predictions.png', dpi=150, bbox_inches='tight')

# %%
# 3dplot!
fig = plt.figure(figsize=(10, 4))

# Make data.
y = np.arange(lat_y)
yy, tt = np.meshgrid(y, t_hist)

# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
ax.plot_surface(
    yy,
    tt,
    amplitude_hist_1 / epsilon,
    cmap=cm.get_cmap('coolwarm'),
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True)
ax.set_title('$\omega={}$'.format(omega1))
ax.set_xlabel('$y$')
ax.set_ylabel('$t$')
ax.set_zlabel('Amplitude')

# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
ax.plot_surface(
    yy,
    tt,
    amplitude_hist_2 / epsilon,
    cmap=cm.get_cmap('coolwarm'),
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True)
ax.set_title('$\omega={}$'.format(omega2))
ax.set_xlabel('$y$')
ax.set_ylabel('$t$')
ax.set_zlabel('Amplitude')

plt.tight_layout()
plt.savefig('./plots/shearwave_time.png', dpi=150, bbox_inches='tight')

#%%
print("Plotting complete. Results saved in ./plots/")
