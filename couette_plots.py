import os
import gzip

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import _pickle as pickle

file_list = ['couette', 'couette_v', 'cavity', 'cavity_v']

res = [pickle.load(gzip.open(os.path.join('pickles', f + '.pkl.gz'), 'rb')) for f in file_list]

# data that should be the same across pickles
x = np.arange(res[0]['lat_x'])
y = np.arange(res[0]['lat_y'])
t = res[0]['t_hist']
yy, tt = np.meshgrid(y, t)

plt.rcParams.update(plt.rcParamsDefault)

#produce charts
for ri, r in enumerate(res):

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

        ax[i].scatter(*r['walls'].T, marker='s', s=1, color='red')

        ax[i].streamplot(
            x,
            y,
            *np.transpose(r['flow_hist'][i], [2, 1, 0]),
            linewidth=(300) * np.linalg.norm(r['flow_hist'][i], axis=2).T)

    plt.savefig(
        './plots/{}.png'.format(file_list[ri]),
        dpi=150,
        bbox_inches='tight')

print("Plotting complete. Results saved in ./plots/")
