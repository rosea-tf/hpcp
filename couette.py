"""
Simulation of couette flow

@author: AlexR
"""
#%% SETUP

import numpy as np
from mpi4py import MPI
from lattice import Lattice
import matplotlib.pyplot as plt
import os
import _pickle as pickle
import gzip
import time
from utils import fetch_dim_args, pickle_save

#%% SET PARAMETERS

[lat_x, lat_y], grid_dims = fetch_dim_args(lat_default=[400, 300])

omega = 1.0

t_recordpoints = [
    100, 150, 200, 400, 700, 1000, 2000, 3000, 4000, 6000, 8000, 10000
]

max_timesteps = max(t_recordpoints)

# lid velocity
ux_lid = 0.01
uy_lids = [0, 0.0001]
uy_period = 100

wall_fn_couette = lambda x, y: np.logical_or(y == 0, y == lat_y - 1)
wall_fn_cavity = lambda x, y: np.logical_or.reduce(
    [y == 0, y == lat_y - 1, x == 0, x == lat_x - 1])

wall_fns = [wall_fn_couette, wall_fn_cavity]
outfiles = ['couette.pkl.gz', 'cavity.pkl.gz']

#%% SETUP

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# run once without side walls, and once with them
for method in [0, 1]:

    wall_fn = wall_fns[method]
    outfile = outfiles[method]

    # set up the lattice
    lat = Lattice([lat_x, lat_y], grid_dims=grid_dims, wall_fn=wall_fn)

    # this will only be used by cells on the top of the grid
    top_wall = lat.core[:, -1:, :]

    if rank == 0:
        lat.print_info()

    # first dimension will cover the two methods (slide, vibrate)
    flow_hists = np.empty([2, len(t_recordpoints), lat_x, lat_y, 2])

    u_lid = np.array([ux_lid, 0])

    #%% SIMULATION
    for i_uy, uy_lid in enumerate(uy_lids):

        lat.reset_to_eq()
        flow_hist = flow_hists[i_uy]

        for t in range(max_timesteps + 1):
            lat.halo_copy()
            lat.stream()  #bounceback occurs in here

            # if it sits at the topmost edge, add sliding lid effect
            # bounceback has already occurred, so rho_wall will be calculated on 2x(7,4,8) instead of 2x(6,2,5)
            if lat.cart.coords[1] == lat.grid_dims[1] - 1:

                # sine-wave wobble for the lid (if any)
                u_lid[1] = uy_lid * np.sin(2 * np.pi * t / uy_period)

                rho_wall = np.sum(
                    top_wall[:, :, [0, 3, 1]] +
                    (2 * top_wall[:, :, [7, 4, 8]]),
                    axis=2,
                    keepdims=True)

                drag = 6 * lat.W * rho_wall * np.einsum(
                    'id,d->i', lat.C, u_lid)

                top_wall[:, :, [7, 4, 8]] += drag[:, :, [7, 4, 8]]

            lat.collide(omega=omega)

            # record entire u lattice
            if t in t_recordpoints:
                u_snapshot = lat.gather(lat.u())

                if rank == 0:
                    flow_hist[t_recordpoints.index(t)] = u_snapshot

    #%% SAVE TO FILE
    if rank == 0:

        # reconstruct walls
        walls = np.array([[x, y] for x in np.arange(lat_x)
                          for y in np.arange(lat_y) if wall_fn(x, y)])

        d = dict(((k, eval(k)) for k in [
            'lat_x', 'lat_y', 'omega', 't_recordpoints', 'flow_hists', 'walls'
        ]))

        pickle_save(outfile, d)
