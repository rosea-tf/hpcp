"""
 Simulate Poiseuille flow (channel periodic in the x-axis, with top and bottom walls, and constant inflow at the left)

ARGUMENTS
    --lat_x, --lat_y : int
        overrides default x- and y-dimensions of the entire lattice

    --grid_x, --grid_y : int
        overrides default x- and y-dimensions of the cartesian processor arrangement. If specified, their product must match the number of processors in use.

INPUTS
    None

OUTPUTS
    poiseuille.pkl.gz
        Compressed results file. Used for plotting.

@author: AlexR
"""

#%% IMPORTS

import numpy as np
from mpi4py import MPI

import _pickle as pickle
from lattice import Lattice
from utils import fetch_dim_args, pickle_save

#%% SET PARAMETERS
[lat_x, lat_y], grid_dims = fetch_dim_args(lat_default=[1000, 100])
timesteps = 10000
interval_hf = 50  # between recordings of flow at halfway point
interval_sp = 200  # between recordings for streamplot
maxints_sp = 9  # number of streamplot frames to record
omega = 1.0

# prescribed inflow at the LHS
inflow = 0.01

# walls (a row of dry cells) at top and bottom
wall_fn = lambda x, y: np.logical_or(y == 0, y == lat_y - 1)

outfile = 'poiseuille.pkl.gz'

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

# set up container for data used in the analysis of flow at the halfway point
halfway_vel_hist = np.empty([timesteps // interval_hf, lat_y, 2])

# set up container for data used in the streamplot
flow_hist = np.empty([maxints_sp, lat_x, lat_y, 2])

#%% SIMULATION

for t in range(timesteps):
    lat.halo_copy()
    lat.stream()

    # get velocity from our cell
    u = lat.u()

    # if it sits at the leftmost edge, prescribe its inflow
    if lat.cart.coords[0] == 0:
        u[0, :, 0] = inflow  # [left, all, x-direction]

    lat.collide(omega=omega, u=u)

    # record flow at halfway point
    if t % interval_hf == 0:
        u_snapshot = lat.gather(lat.u())

        if rank == 0:
            # collect data from "monitoring station" halfway down the channel
            halfway_vel_hist[t // interval_hf] = u_snapshot[lat_x // 2]

    # record entire u lattice
    if t % interval_sp == 0 and t <= t_hist_sp[-1]:
        u_snapshot = lat.gather(lat.u())
        if rank == 0:
            flow_hist[t // interval_sp] = u_snapshot

#%% SAVE RESULTS
if rank == 0:
    # reconstruct walls
    walls = np.array([[x, y] for x in np.arange(lat_x)
                      for y in np.arange(lat_y) if wall_fn(x, y)])

    d = dict(((k, eval(k)) for k in [
        'lat_x', 'lat_y', 'inflow', 'omega', 't_hist_hf', 't_hist_sp',
        'halfway_vel_hist', 'flow_hist', 'walls'
    ]))

    pickle_save(outfile, d)
