#%% CHARTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import argparse
import _pickle as pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "results1", type=str, help="filename of first pickled results set")
args = parser.parse_args()

# args.results1 = 'sinedensity.pkl'

res1 = pickle.load(open(os.path.join('pickles', args.results1), 'rb'))

# make data
x = np.arange(res1['lat_x'])
t = res1['t_hist']
xx, tt = np.meshgrid(x, t)

plt.rcParams.update(plt.rcParamsDefault)

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
    res1['density_hist'],
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
    res1['velocity_hist'],
    rstride=1,
    cstride=1,
    cmap=cm.get_cmap('coolwarm'),
    linewidth=0,
    antialiased=True)

plt.tight_layout()
plt.savefig('./plots/sinedensity.png', dpi=150, bbox_inches='tight')

print("Plotting complete. Results saved in ./plots/")
