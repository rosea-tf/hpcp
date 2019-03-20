#%% CHARTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import argparse
import _pickle as pickle
import os

res1 = pickle.load(open(os.path.join('pickles', 'couette.pkl'), 'rb'))
res2 = pickle.load(open(os.path.join('pickles', 'couette_vibrate.pkl'), 'rb'))

# load data from pickles
x = np.arange(res1['lat_x'])
y = np.arange(res1['lat_y'])
t = res1['t_hist']
yy, tt = np.meshgrid(y, t)
walls = res1['walls']
flow_hist = [res1['flow_hist'], res2['flow_hist']]

plt.rcParams.update(plt.rcParamsDefault)

#produce charts
#i = 0: no wobble
#i = 1: wobble
for r in [0, 1]:

    fig, axc = plt.subplots(3, 3, sharex=True, sharey=True, figsize=[10, 8])

    for a in axc[-1]:
        a.set_xlabel('$x$')
    for a in axc[:, 0]:
        a.set_ylabel('$y$')

    ax = axc.reshape(-1)
    for i in range(9):
        ax[i].set_title('t={}'.format(t[i]))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        ax[i].scatter(*walls.T, marker='s', s=1, color='red')

        ax[i].streamplot(
            x,
            y,
            *np.transpose(flow_hist[r][i], [2, 1, 0]),
            linewidth=(300) * np.linalg.norm(flow_hist[r][i], axis=2).T)

    plt.savefig(
        './plots/couette_{}.png'.format(r), dpi=150, bbox_inches='tight')


print("Plotting complete. Results saved in ./plots/")
