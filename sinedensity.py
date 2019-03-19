"""
Simulation of sine-distributed density evolution

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

# %%

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str, help="output filename")
parser.add_argument("lat_x", type=int, help="x-length of lattice to simulate")
parser.add_argument("lat_y", type=int, help="y-length of lattice to simulate")
parser.add_argument("--grid_x", type=int, help="x-length of process grid")
parser.add_argument("--grid_y", type=int, help="y-length of process grid")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument(
    "--interval",
    type=int,
    help="number of timesteps between data recordings",
    default=10)
parser.add_argument("--omega", type=float, default=1.0)
parser.add_argument("--epsilon", type=float, default=0.01)
args = parser.parse_args()


# # %%
# class C:
#     pass


# args = C()
# args.timesteps = 1500
# args.interval = 10
# args.lat_x = 100
# args.lat_y = 80
# args.epsilon = 0.01
# args.grid_x = None
# args.grid_y = None
# args.omega=0.5
# args.output = 'sinedensity.pkl'
# #%%

lat_x = args.lat_x
lat_y = args.lat_y
epsilon = args.epsilon
t_hist = np.arange(args.timesteps, step=args.interval)

grid_dims = [args.grid_x, args.grid_y
             ] if args.grid_x is not None and args.grid_y is not None else None

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up the lattice
lat = Lattice([lat_x, lat_y], grid_dims=grid_dims)

if rank == 0:
    lat.print_info()

lat.reset_to_eq()

# Strength of sine-wave disturbance
sin_x = (epsilon * np.sin(
    2 * np.pi * lat.cell_ranges[0] / lat_x))[:, np.newaxis, np.newaxis]

rho_modified = lat.rho() + sin_x

#density and velocity are constant wrt y, so we only need to store one row of x at every timestep
density_hist = np.empty([args.timesteps // args.interval, lat_x])
velocity_hist = np.empty([args.timesteps // args.interval,
                          lat_x])  #storing x-velocity only

# run collision operator once, feeding this prescribed rho in
lat.collide(rho=rho_modified)

#%% SIMULATION

for t in range(args.timesteps):
    lat.halo_copy()
    lat.stream()
    lat.collide(omega=args.omega)

    if t % args.interval == 0:
        # save a plot
        rho_snapshot = lat.gather(lat.rho())
        u_snapshot = lat.gather(lat.u())

        if rank == 0:
            density_hist[t // args.interval] = rho_snapshot[:, 0, 0]
            velocity_hist[t // args.interval] = u_snapshot[:, 0, 0]

            # sine wave pattern should be consistent across all y

            assert np.isclose(rho_snapshot, rho_snapshot[:, 0:1]).all()

            assert np.isclose(u_snapshot, u_snapshot[:, 0:1]).all()
    

if rank == 0:
    # save variables and exit
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    d = dict(
        ((k, eval(k)) for k in
         ['lat_x','lat_y', 'epsilon', 't_hist', 'density_hist', 'velocity_hist']))

    pickle.dump(d, open(os.path.join(pickle_path, args.output), 'wb'))

    print("Results saved to ./pickles/{}".format(args.output))


#%%
