#%%
import os
import gzip

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import _pickle as pickle

# %% LOAD RESULTS

counts = [1, 2, 4, 8, 16, 32, 56, 112]
methods = ['2D Grid', '1D Grid', '4x Size, 0.25x Time']

MLUP = 40 * 30 * 200

res = {
    c: pickle.load(
        gzip.open(os.path.join('pickles', 'time_{}.pkl.gz'.format(c)), 'rb'))
    for c in counts
}

# %%

plt.rcParams.update(plt.rcParamsDefault)

plt.clf()
fig, axc = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 8))

ax = axc.flatten()

ax[0].set_title('Wall Clock Time')
# ax[0].set_xlabel('Number of Processors')
ax[0].set_ylabel('Seconds')

ax[1].set_title('CPU Communication Time')
# ax[1].set_xlabel('Number of Processors')
# ax[1].set_ylabel('Seconds')

ax[2].set_title('CPU Computation Time')
ax[2].set_xlabel('Number of Processors')
ax[2].set_ylabel('Seconds')

ax[3].set_title('Total CPU Time')
ax[3].set_xlabel('Number of Processors')
# ax[3].set_ylabel('Seconds')

for im, method in enumerate(methods):

    # wall time
    t_wall = [res[c][im, :, 1].max() - res[c][im, :, 0].min() for c in counts]

    t_comp = [res[c][im, :, 2].sum() for c in counts]

    t_comm = [res[c][im, :, 3].sum() for c in counts]

    t_total = [m + p for m, p in zip(t_comm, t_comp)]

    ax[0].plot(counts, t_wall, '-o', label=method)
    ax[1].plot(counts, t_comm, '-o', label=method)
    ax[2].plot(counts, t_comp, '-o', label=method)
    ax[3].plot(counts, t_total, '-o', label=method)

for a in ax:
    a.legend()

fig.savefig('./plots/time.png', dpi=150, bbox_inches='tight')
