"""
Simulation of couette flow

@author: AlexR
"""
#%% SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from mpi4py import MPI
from lattice import Lattice
import matplotlib.pyplot as plt
import argparse
import os
import _pickle as pickle

# parser = argparse.ArgumentParser()
# parser.add_argument("output", type=str, help="output filename")
# parser.add_argument("lat_x", type=int, help="x-length of lattice to simulate")
# parser.add_argument("lat_y", type=int, help="y-length of lattice to simulate")
# parser.add_argument("--grid_x", type=int, help="x-length of process grid")
# parser.add_argument("--grid_y", type=int, help="y-length of process grid")
# parser.add_argument("--timesteps", type=int, default=1000)
# parser.add_argument(
#     "--interval",
#     type=int,
#     help="number of timesteps between data recordings",
#     default=10)
# parser.add_argument("--omega", type=float, default=1.0)
# parser.add_argument("--epsilon", type=float, default=0.01)
# args = parser.parse_args()


# %%
class C:
    pass

args = C()
args.timesteps = 500
args.interval = 50
args.lat_x = 100
args.lat_y = 80
args.grid_x = None
args.grid_y = None
args.omega = 1.0
args.lidvelocity = 0.01
args.output = 'couette.pkl'

lat_x = args.lat_x
lat_y = args.lat_y
lidvelocity = np.array([args.lidvelocity, 0])
omega = args.omega
t_hist = np.arange(args.timesteps, step=args.interval)

grid_dims = [args.grid_x, args.grid_y
             ] if args.grid_x is not None and args.grid_y is not None else None

# walls (a row of dry cells) at top and bottom
wall_fn = lambda x, y: np.logical_or(y == 0, y == lat_y - 1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up the lattice
lat = Lattice([lat_x, lat_y], grid_dims=grid_dims, wall_fn=wall_fn)

if rank == 0:
    lat.print_info()


flow_hist = np.empty([args.timesteps // args.interval, lat_x, lat_y, 2])

#%% SIMULATION

for t in range(args.timesteps):
    lat.halo_copy()
    lat.stream()
    lat.collide(omega=omega)

    # if it sits at the topmost edge, add sliding lid effect
    # bounceback has already occurred, so rho_wall will be calculated on 2x(7,4,8) instead of 2x(6,2,5)
    if lat.cart.coords[1] == lat.grid_dims[1] - 1:
        
        # good
        # lidvelocity = np.array([args.lidvelocity, 0.001 * np.sin(2 * np.pi * t / 100)])

        top_wall = lat.core[:, -1:, :]

        rho_wall = np.sum(top_wall[:, :, [0, 3, 1]] + (2 * top_wall[:, :, [7, 4, 8]]), axis=2, keepdims=True)  

        drag = 6 * lat.W * rho_wall * np.einsum('id,d->i', lat.C, lidvelocity)

        top_wall[:, :, [7, 4, 8]] += drag[:, :, [7, 4, 8]] #not minus?

    if t % args.interval == 0:
        u_snapshot = lat.gather(lat.u())

        if rank == 0:
            flow_hist[t // args.interval] = u_snapshot


# %%
if rank == 0: #TODO
    # reconstruct walls
    walls = np.array([[x, y] for x in np.arange(lat_x) for y in np.arange(lat_y)
                    if wall_fn(x, y)])

    # save variables and exit
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    d = dict(((k, eval(k)) for k in [
        'lat_x', 'lat_y', 'lidvelocity', 'omega', 't_hist', 'flow_hist', 'walls'
    ]))

    pickle.dump(d, open(os.path.join(pickle_path, args.output), 'wb'))

    print("Results saved to ./pickles/{}".format(args.output))
