"""
Plotting script to be run after oscillator.py

Serial only.

@author: AlexR
"""
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
    gzip.open(os.path.join('pickles', 'oscillator.pkl.gz'), 'rb'))

lat_x = res['lat_x']
lat_y = res['lat_y']
omega = res['omega']
walls = res['walls']
inflow = res['inflow']
flow_hist = res['flow_hist']
t_recordpoints = res['t_recordpoints']

x = np.arange(lat_x)
y = np.arange(lat_y)

plt.rcParams.update(plt.rcParamsDefault)

#%% STREAMPLOTS
plt.clf()
x = np.arange(lat_x)
y = np.arange(lat_y)

fig, axc = plt.subplots(3, 3, sharex=True, sharey=True, figsize=[10, 8])

for a in axc[-1]:
    a.set_xlabel('$x$')
for a in axc[:, 0]:
    a.set_ylabel('$y$')

ax = axc.reshape(-1)

for i in range(9):
    ax[i].set_title('t={}'.format(t_recordpoints[i]))
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    ax[i].imshow(walls.T, cmap=cm.get_cmap('Pastel2'))
    ax[i].streamplot(
        x,
        y,
        *np.transpose(flow_hist[i], [2, 1, 0]),
        linewidth=(100) * np.linalg.norm(flow_hist[i], axis=2).T)

plot_save(fig, 'oscillator.png')
