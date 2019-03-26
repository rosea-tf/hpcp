import os
import gzip

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import _pickle as pickle

plt.rcParams.update(plt.rcParamsDefault)

methods = ['couette', 'cavity']

uy_description = ['Sliding lid', 'Sliding/Vibrating Lid']

# iterate over both ex 5 and 6 results
for method in methods:

    res = pickle.load(
        gzip.open(os.path.join('pickles', method + '.pkl.gz'), 'rb'))

    lat_x = res['lat_x']
    lat_y = res['lat_y']

    walls = res['walls']

    x = np.arange(lat_x)
    y = np.arange(lat_y)

    # t_hist_sp = res['t_hist_sp']
    # t_hist_hf = res['t_hist_hf']
    # halfway_vel_hists = res['halfway_vel_hists']
    t_recordpoints = res['t_recordpoints']
    flow_hists = res['flow_hists']

    # yy, tt = np.meshgrid(y, t_hist_hf)

    fig2d = plt.figure(figsize=[10, 4])

    # iterate over lid stable / wobbling
    for i_uy in [0, 1]:
        halfway_vel_final = flow_hists[i_uy, -1, lat_x // 2]
        flow_hist = flow_hists[i_uy]

        # add a panel to 2d chart
        ax2d = fig2d.add_subplot(1, 2, 1 + i_uy)
        ax2d.set_title(uy_description[i_uy])
        ax2d.plot(halfway_vel_final[:-1, 0],
                  y[:-1])  #cut out the last cell, which is the dry cell
        # and plot the x-velocity only
        ax2d.set_ylabel('$y$')
        ax2d.set_xlabel('$u_x$')
        ax2d.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # stream plot
        fig, axc = plt.subplots(
            4, 3, sharex=True, sharey=True, figsize=[10, 10])

        for a in axc[-1]:
            a.set_xlabel('$x$')
        for a in axc[:, 0]:
            a.set_ylabel('$y$')

        ax = axc.reshape(-1)
        for i in range(len(t_recordpoints)):
            ax[i].set_title('t={}'.format(t_recordpoints[i]))
            ax[i].set_xticks([])
            ax[i].set_yticks([])

            ax[i].scatter(*walls.T, marker='s', s=1, color='red')

            ax[i].streamplot(
                x,
                y,
                *np.transpose(flow_hist[i], [2, 1, 0]),
                linewidth=(300) * np.linalg.norm(flow_hist[i], axis=2).T)

        fig.savefig(
            './plots/{}_stream_{}.png'.format(method, i_uy),
            dpi=150,
            bbox_inches='tight')

        # zoomed-in stream plot
        fig, axc = plt.subplots(
            4, 3, sharex=True, sharey=True, figsize=[10, 8])

        for a in axc[-1]:
            a.set_xlabel('$x$')
        for a in axc[:, 0]:
            a.set_ylabel('$y$')

        ax = axc.reshape(-1)

        for i in range(len(t_recordpoints)):

            flow_zoom = flow_hist[i, :, int(lat_y * 0.8):]

            ax[i].set_title('t={}'.format(t_recordpoints[i]))
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_ylim(bottom=int(lat_y * 0.8), top=lat_y)

            ax[i].scatter(*walls.T, marker='s', s=1, color='red')

            ax[i].streamplot(
                x,
                y[int(lat_y * 0.8):],
                *np.transpose(flow_zoom, [2, 1, 0]),
                linewidth=(100) * np.linalg.norm(flow_zoom, axis=2).T)

        fig.savefig(
            './plots/{}_streamzoom_{}.png'.format(method, i_uy),
            dpi=150,
            bbox_inches='tight')

    fig2d.savefig(
        './plots/{}_halfway.png'.format(method), dpi=150, bbox_inches='tight')

print("Plotting complete. Results saved in ./plots/")
