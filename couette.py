"""
Simulation of couette flow

@author: AlexR
"""
#%% SETUP

import numpy as np
from mpi4py import MPI
from lattice import Lattice
import matplotlib.pyplot as plt
import argparse
import os
import _pickle as pickle
import gzip
import time

#%% PARSE COMMAND LINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str, help="output filename")
parser.add_argument("lat_x", type=int, help="x-length of lattice to simulate")
parser.add_argument("lat_y", type=int, help="y-length of lattice to simulate")
parser.add_argument("--grid_x", type=int, help="x-length of process grid")
parser.add_argument("--grid_y", type=int, help="y-length of process grid")
parser.add_argument("--timesteps", type=int, default=500)
parser.add_argument(
    "--interval",
    type=int,
    help="number of timesteps between data recordings",
    default=50)
parser.add_argument("--omega", type=float, default=1.0)

parser.add_argument(
    "--wall_x",
    dest='wall_x',
    action='store_true',
    help="creates walls at sides (i.e. a box)")
parser.add_argument(
    "--ux_lid",
    type=float,
    default=0.01,
    help="horizontal speed of sliding lid")
parser.add_argument(
    "--uy_lid_period",
    type=float,
    default=0,
    help="period (timesteps) of sine wave wobble for lid (0 for none)")
parser.add_argument(
    "--timeit",
    dest='timeit',
    action='store_true',
    help="outputs time taken (rather than lattice data)")

args = parser.parse_args()

# %% SETUP

t0 = time.time()
t_copy_total = 0.0

UY_STRENGTH = 0.1

lat_x = args.lat_x
lat_y = args.lat_y
omega = args.omega
t_hist = np.arange(args.timesteps, step=args.interval)

grid_dims = [args.grid_x, args.grid_y
             ] if args.grid_x is not None and args.grid_y is not None else None

if not args.wall_x:
    # walls (a row of dry cells) at top and bottom
    wall_fn = lambda x, y: np.logical_or(y == 0, y == lat_y - 1)
else:
    # optionally, walls at the sides as well
    wall_fn = lambda x, y:  np.logical_or.reduce(
        [y == 0, y == lat_y - 1, x == 0, x == lat_x - 1])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up the lattice
lat = Lattice([lat_x, lat_y], grid_dims=grid_dims, wall_fn=wall_fn)

# this will only be used by cells on the top of the grid
top_wall = lat.core[:, -1:, :]

if rank == 0:
    lat.print_info()

flow_hist = np.empty([args.timesteps // args.interval, lat_x, lat_y, 2])

#%% SIMULATION

for t in range(args.timesteps):
    t_precopy = time.time()
    lat.halo_copy()
    t_copy_total += (time.time() - t_precopy)

    lat.stream()
    lat.collide(omega=omega)

    # if it sits at the topmost edge, add sliding lid effect
    # bounceback has already occurred, so rho_wall will be calculated on 2x(7,4,8) instead of 2x(6,2,5)
    if lat.cart.coords[1] == lat.grid_dims[1] - 1:

        # sine-wave wobble for the lid (if any)
        uy_lid = 0 if args.uy_lid_period == 0 else args.ux_lid * UY_STRENGTH * np.sin(
            2 * np.pi * t / args.uy_lid_period)

        # calculate final lid velocity
        u_lid = np.array([
            args.ux_lid,
        ])

        rho_wall = np.sum(
            top_wall[:, :, [0, 3, 1]] + (2 * top_wall[:, :, [7, 4, 8]]),
            axis=2,
            keepdims=True)

        drag = 6 * lat.W * rho_wall * np.einsum('id,d->i', lat.C, u_lid)

        top_wall[:, :, [7, 4, 8]] += drag[:, :, [7, 4, 8]]

    # record data
    if t % args.interval == 0 and not args.timeit:
        u_snapshot = lat.gather(lat.u())

        if rank == 0:
            flow_hist[t // args.interval] = u_snapshot

t_total = (time.time() - t0)

if args.timeit:
    t_gather = np.empty([size, 2])
    comm.Gather(np.array([t_total, t_copy_total]), t_gather, root=0)

#%% SAVE TO FILE
if rank == 0:
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
    
    if not args.timeit:
        # reconstruct walls
        walls = np.array([[x, y] for x in np.arange(lat_x)
                        for y in np.arange(lat_y) if wall_fn(x, y)])

        d = dict(
            ((k, eval(k))
            for k in ['lat_x', 'lat_y', 'omega', 't_hist', 'flow_hist', 'walls']))

        pickle.dump(d, gzip.open(os.path.join(pickle_path, args.output + '.pkl.gz'), 'wb'))

        print("Results saved to ./pickles/{}.pkl.gz".format(args.output))
    
    else:
        pickle.dump(t_gather, open(os.path.join(pickle_path, args.output + '_time.pkl'), 'wb'))
        print("Results saved to ./pickles/{}_time.pkl".format(args.output))
        
