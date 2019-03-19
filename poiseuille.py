"""
Simulation of poseuille flow

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
args.timesteps = 1000
args.interval = 5
args.lat_x = 100
args.lat_y = 80
args.grid_x = None
args.grid_y = None
args.omega = 1.0
args.inflow = 0.01
args.output = 'poseuille.pkl'

lat_x = args.lat_x
lat_y = args.lat_y
inflow = args.inflow
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

halfway_vel_hist = np.empty([args.timesteps // args.interval, lat_y, 2])

flow_hist = np.empty([10, lat_x, lat_y, 2])
flow_recordpoints = np.arange(
    args.timesteps / (20 / 9), step=args.timesteps / 20, dtype=np.int)

#%% SIMULATION

for t in range(args.timesteps):
    lat.halo_copy()
    lat.stream()

    # get velocity from our cell
    u = lat.u()

    # if it sits at the leftmost edge, prescribe its inflow
    if lat.cart.coords[0] == 0:
        u[0, :, 0] = inflow
        u[0, :, 1] = 0  # ???

    lat.collide(omega=args.omega, u=u)

    # record flow at halfway point
    if t % args.interval == 0:
        u_snapshot = lat.gather(lat.u())

        if rank == 0:
            # collect data from "monitoring station" halfway down the channel
            halfway_vel_hist[t // args.interval] = u_snapshot[lat_x // 2]

    if np.isin(t, flow_recordpoints):
        u_snapshot = lat.gather(lat.u())
        flow_hist[np.where(flow_recordpoints == t)] = u_snapshot


# %%
if rank == 0:
    # reconstruct walls
    walls = np.array([[x, y] for x in np.arange(lat_x) for y in np.arange(lat_y)
                    if wall_fn(x, y)])

    # save variables and exit
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    d = dict(((k, eval(k)) for k in [
        'lat_x', 'lat_y', 'inflow', 'omega', 't_hist', 'halfway_vel_hist',
        'flow_hist', 'flow_recordpoints', 'walls'
    ]))

    pickle.dump(d, open(os.path.join(pickle_path, args.output), 'wb'))

    print("Results saved to ./pickles/{}".format(args.output))
