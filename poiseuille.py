"""
Simulation of poseuille flow

@author: AlexR
"""
#%% IMPORTS

import numpy as np
from mpi4py import MPI
from lattice import Lattice
import matplotlib.pyplot as plt
import os
import _pickle as pickle
import gzip
from utils import fetch_grid_dims

#%% SET PARAMETERS
lat_x = 400
lat_y = 300
timesteps = 5000  #TODO
interval_hf = 5  #between recordings of flow at halfway point
interval_sp = 50  #between recordings for streamplot
maxints_sp = 9  # number of streamplot frames to record

omega = 1.0

# prescribed inflow at the LHS
inflow = 0.01
# walls (a row of dry cells) at top and bottom
wall_fn = lambda x, y: np.logical_or(y == 0, y == lat_y - 1)

outfile = 'poiseuille.pkl.gz'

grid_dims = fetch_grid_dims()

#%% SETUP
t_hist_hf = np.arange(timesteps, step=interval_hf)
t_hist_sp = np.arange(interval_sp * maxints_sp, step=interval_sp)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up the lattice
lat = Lattice([lat_x, lat_y], grid_dims=grid_dims, wall_fn=wall_fn)

if rank == 0:
    lat.print_info()

halfway_vel_hist = np.empty([timesteps // interval_hf, lat_y, 2])

flow_hist = np.empty([maxints_sp, lat_x, lat_y, 2])

#%% SIMULATION

for t in range(timesteps):
    lat.halo_copy()
    lat.stream()

    # get velocity from our cell
    u = lat.u()

    # if it sits at the leftmost edge, prescribe its inflow
    if lat.cart.coords[0] == 0:
        u[0, :, 0] = inflow
        u[0, :, 1] = 0  # ???

    lat.collide(omega=omega, u=u)

    # record flow at halfway point
    if t % interval_hf == 0:
        u_snapshot = lat.gather(lat.u())

        if rank == 0:
            # collect data from "monitoring station" halfway down the channel
            halfway_vel_hist[t // interval_hf] = u_snapshot[lat_x // 2]

    # record entire u lattice
    if t % interval_sp == 0 and t < t_hist_sp[-1]:
        u_snapshot = lat.gather(lat.u())
        if rank == 0:
            flow_hist[t // interval_sp] = u_snapshot

#%% SAVE AND EXIT
if rank == 0:
    # reconstruct walls
    walls = np.array([[x, y] for x in np.arange(lat_x)
                      for y in np.arange(lat_y) if wall_fn(x, y)])

    # save variables and exit
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    d = dict(((k, eval(k)) for k in [
        'lat_x', 'lat_y', 'inflow', 'omega', 't_hist_hf', 't_hist_sp',
        'halfway_vel_hist', 'flow_hist', 'walls'
    ]))

    outpath = os.path.join(pickle_path, outfile)

    pickle.dump(d, gzip.open(outpath, 'wb'))

    print("Results saved to " + outpath)

#%%
