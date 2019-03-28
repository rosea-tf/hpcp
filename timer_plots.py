"""
Plotting script to be run after timer.py

Serial only.

@author: AlexR
"""
#%% IMPORTS

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import plot_save

# %% LOAD RESULTS

counts = [1, 2, 4, 8, 16, 32, 56, 112, 224]
methods = ['2D Grid', '1D Grid', '4x Size, 0.25x Time']
MLUP = 400 * 300 * 2000 / 1e6


def text_to_time(line):
    timesplt = line.split('m')
    mins = float(timesplt[0])
    secs = float(timesplt[1])
    total = mins * 60 + secs
    return total


results = {}

for ic, c in enumerate(counts):

    results[c] = {}

    with open('output/timer{}.txt'.format(c)) as fh:
        lines = fh.readlines()

    # parse the output of the unix 'time' command
    method = -1
    for line in lines:
        if line[:7] == 'METHOD ':
            method = int(line[7:])
            results[c][method] = {}
        if line[:5] == 'real\t':
            total = text_to_time(line[5:-2])  #exclude 's'
            results[c][method]['w'] = total
        if line[:5] == 'user\t':
            total = text_to_time(line[5:-2])  #exclude 's'
            results[c][method]['u'] = total
        if line[:4] == 'sys\t':
            total = text_to_time(line[4:-2])  #exclude 's'
            results[c][method]['s'] = total

#%% PLOTS

plt.rcParams.update(plt.rcParamsDefault)

plt.clf()

fig_secs, ax_secs = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
ax_secs[0].set_title('Wall Clock Time')
ax_secs[0].set_ylabel('Seconds')

ax_secs[1].set_title('CPU Time')

fig_mlupl, ax_mlupl = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
ax_mlupl[0].set_title('Lattice Update Frequency (Wall Clock)')
ax_mlupl[0].set_ylabel('MLUPS')

ax_mlupl[1].set_title('Lattice Update Frequency (CPU Time)')

for im, method in enumerate(methods):

    # wall time
    t_wall = np.array([results[c][im]['w'] for c in counts])

    t_cpu = np.array(
        [results[c][im]['u'] + results[c][im]['s'] for c in counts])

    ax_secs[0].plot(counts, t_wall, '-o', label=method)
    ax_secs[1].plot(counts, t_cpu, '-o', label=method)
    ax_mlupl[0].plot(counts, MLUP / t_wall, '-o', label=method)
    ax_mlupl[1].plot(counts, MLUP / t_cpu, '-o', label=method)

# for every axes...
for a in ax_secs:
    a.set_xlabel('Number of Processors')
    a.legend()

for a in ax_mlupl:
    a.set_xlabel('Number of Processors')
    a.legend()
    a.set_xscale("log")
    a.set_yscale("log")

plot_save(fig_secs, 'time_secs.png')
plot_save(fig_mlupl, 'time_mlupl.png')

#%% FIT AMDAHL'S LAW

p = np.array(counts)
y = t_wall / t_wall[0]


def f(p, fp):
    return (1 - fp) + (fp / p)


popt, pcov = curve_fit(f, p, y)

print("Estimated p_f (fraction of parallelisable code): {} (Variance: {})".
      format(popt[0], pcov[0, 0]))
