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
import argparse
import os
import _pickle as pickle
import gzip
import PIL
from PIL import Image

#%% DEFAULT PARAMETERS

interval = 1
lat_x = 400
lat_y = 300
grid_x = None
grid_y = None
omega = 1.0
inflow = 0.01
outfile = 'oscillator.pkl.gz'
t_recordpoints = [
    100, 200, 300, 400, 600, 800, 1000, 2000, 3000
]

#%% PARSE COMMAND LINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument("--lat_x", type=int, help="x-length of lattice to simulate")
parser.add_argument("--lat_y", type=int, help="y-length of lattice to simulate")
parser.add_argument("--grid_x", type=int, help="x-length of process grid")
parser.add_argument("--grid_y", type=int, help="y-length of process grid")
parser.add_argument("--omega", type=float, default=1.0)
args = parser.parse_args()

if args.lat_x is not None:
    lat_x = args.lat_x

if args.lat_y is not None:
    lat_y = args.lat_y

if args.grid_x is not None:
    grid_x = args.grid_x

if args.grid_y is not None:
    grid_y = args.grid_y

if args.omega is not None:
    omega = args.omega

#%% SETUP

# use the figure we cut out of the exercise sheet (a monochrome bitmap)
im = Image.open("oscillator.bmp")

# resize it to desired lattice dimensions
walls = np.asarray(im.resize([lat_x, lat_y])).T

grid_dims = [grid_x, grid_y] if grid_x is not None and grid_y is not None else None

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
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    d = dict(
        ((k, eval(k))
         for k in ['lat_x', 'lat_y', 'omega', 'inflow', 't_recordpoints', 'flow_hist', 'walls']))

    outpath = os.path.join(pickle_path, outfile)

    pickle.dump(d, gzip.open(outpath, 'wb'))

    print("Results saved to " + outpath)

