import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

x = np.arange(10)
y_wet = np.arange(1, 5)
y_dry = np.array([0, 5])

plt.scatter(*np.meshgrid(x, y_wet), c='mediumaquamarine', label='Wet cells')
plt.scatter(*np.meshgrid(x, y_dry), c='grey', label='Dry cells')
plt.xlim(-0.5, 9.5)
plt.hlines(
    y=[0.5, 4.5], xmin=-0.5, xmax=9.5, colors='burlywood', label='Walls')
plt.legend()
plt.xticks([])
plt.yticks([])
plt.savefig('./plots/walls.png', dpi=150, bbox_inches='tight')
