"""
Experiment with the effect that processing node count has on the
time taken to compute the Lid-Driven Cavity experiment of Chapter 7. 

@author: AlexR
"""
#%% SETUP

import numpy as np
from mpi4py import MPI
from lattice import Lattice
import matplotlib.pyplot as plt
from utils import pickle_save
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument(
    "method",
    type=int,
    choices=[0, 1, 2],
    help="0: normal; 1: 1-dimensional grid; 2: larger grid, shorter time")
args = parser.parse_args()

method = args.method

#%% SET PARAMETERS

# dimensions, grid layout, and timesteps
# for each of the three methods we'll be trying
lat_x_s = [400, 400, 800]
lat_y_s = [300, 300, 600]
timesteps_s = [2000, 2000, 500]
grid_dims_s = [None, [size, 1], None]

omega = 1.0

# lid velocity - constant, no vibrations
u_lid = np.array([0.01, 0])

# we're in a box, now.
wall_fn = lambda x, y: np.logical_or.reduce(
    [y == 0, y == lat_y - 1, x == 0, x == lat_x - 1])

# this will be coded with the number of processors in use
outfile = 'time_p{}_m{}.pkl.gz'

#%% SETUP
lat_x = lat_x_s[method]
lat_y = lat_y_s[method]
timesteps = timesteps_s[method]
grid_dims = grid_dims_s[method]

# set up the lattice
lat = Lattice([lat_x, lat_y], grid_dims=grid_dims, wall_fn=wall_fn)

# this will only be used by cells on the top of the grid
top_wall = lat.core[:, -1:, :]

if rank == 0:
    print("METHOD {}".format(method))
    lat.print_info()

#%% SIMULATION
for t in range(timesteps):
    lat.halo_copy()
    lat.stream()
    lat.collide(omega=omega)

    # if it sits at the topmost edge, add sliding lid effect
    # bounceback has already occurred, so rho_wall will be calculated on 2x(7,4,8) instead of 2x(6,2,5)
    if lat.cart.coords[1] == lat.grid_dims[1] - 1:

        rho_wall = np.sum(
            top_wall[:, :, [0, 3, 1]] + (2 * top_wall[:, :, [7, 4, 8]]),
            axis=2,
            keepdims=True)

        drag = 6 * lat.W * rho_wall * np.einsum('id,d->i', lat.C, u_lid)

        top_wall[:, :, [7, 4, 8]] += drag[:, :, [7, 4, 8]]

comm.Gather(
    np.array([t_start, t_end, t_comp, t_copy]), time_results[method], root=0)
