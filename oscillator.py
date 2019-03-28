"""
Simulation of fluidic oscillator

@author: AlexR
"""
#%% IMPORTS

import numpy as np
from mpi4py import MPI
from lattice import Lattice
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import _pickle as pickle
import gzip
import PIL
from PIL import Image
from utils import fetch_dim_args, pickle_save

#%% DEFAULT PARAMETERS

[lat_x, lat_y], grid_dims = fetch_dim_args(lat_default=[400, 300])

interval = 1
omega = 1.0
inflow = 0.01
outfile = 'oscillator.pkl.gz'
t_recordpoints = [
    100, 200, 300, 400, 600, 800, 1000, 2000, 3000
]

#%% SETUP

# use the figure we cut out of the exercise sheet (a monochrome bitmap)
im = Image.open("oscillator.bmp")

# resize it to desired lattice dimensions
walls = np.asarray(im.resize([lat_x, lat_y])).T

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up the lattice
lat = Lattice([lat_x, lat_y], grid_dims=grid_dims)

# set walls directly (this won't work in parallel) # lat.walls = walls
#cut up oscillator pattern into per-node pieces
lat.walls = walls[lat.cell_starts[rank, 0]:lat.cell_starts[rank, 0] +
                     lat.cell_dims[rank, 0], lat.cell_starts[rank, 1]:lat.
                     cell_starts[rank, 1] + lat.cell_dims[rank, 1]]


# find the gap in the left-hand side
inflow_y = np.logical_not(lat.walls[0, :])

# find the gap in the right-hand side
outflow_y = np.logical_not(lat.walls[-1, :])

if rank == 0:
    lat.print_info()

flow_hist = np.empty([len(t_recordpoints), lat_x, lat_y, 2])

max_timesteps = max(t_recordpoints)

#%% SIMULATION

for t in range(max_timesteps + 1):
    lat.halo_copy()
    lat.stream()

    # get velocity from our cell
    u = lat.u()

    # prescribe inflow
    if lat.cart.coords[0] == 0:
        u[0, inflow_y, 0] = inflow
        u[0, inflow_y, 1] = 0

    lat.collide(omega=omega, u=u)

    # record data
    if t in t_recordpoints:
        u_snapshot = lat.gather(lat.u())
        
        # necessary for streamplots to work later
        u_snapshot[walls] = 0

        if rank == 0:
            flow_hist[t_recordpoints.index(t)] = u_snapshot

#%% SAVE TO FILE
if rank == 0:

    d = dict(
        ((k, eval(k))
         for k in ['lat_x', 'lat_y', 'omega', 'inflow', 't_recordpoints', 'flow_hist', 'walls']))

    pickle_save(outfile, d)


