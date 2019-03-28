"""
 Simulate Couette flow (channel periodic in the x-axis, with a moving top wall), for both a smooth and vibrating lid.
 
 Then simulate the lid driven cavity (same thing, but no longer x-periodic)

ARGUMENTS
    --lat_x, --lat_y : int
        overrides default x- and y-dimensions of the entire lattice

    --grid_x, --grid_y : int
        overrides default x- and y-dimensions of the cartesian processor arrangement. If specified, their product must match the number of processors in use.

INPUTS
    None

OUTPUTS
    couette.pkl.gz
        Compressed results file. Used for plotting.
    
    cavity.pkl.gz
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

[lat_x, lat_y], grid_dims = fetch_dim_args(lat_default=[400, 300])

omega = 1.0

# time steps at which to show the streamplot
t_recordpoints = [
    100, 150, 200, 400, 700, 1000, 2000, 3000, 4000, 6000, 8000, 10000
]

max_timesteps = max(t_recordpoints)

# lid velocity: moving to right
ux_lid = 0.01

# two options: smooth sliding, and vibration
uy_lids = [0, 0.0001]

# if vibrating, set the period of the sine wave
uy_period = 100

# walls at top and bottom
wall_fn_couette = lambda x, y: np.logical_or(y == 0, y == lat_y - 1)

# walls at top, bottom, and sides
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

    # set up container for data used in the streamplot
    # first dimension := the two methods (slide, vibrate)
    flow_hists = np.empty([2, len(t_recordpoints), lat_x, lat_y, 2])

    # smooth sliding
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

                # modified bounceback operation...
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

    #%% SAVE RESULTS
    if rank == 0:

        # reconstruct walls
        walls = np.array([[x, y] for x in np.arange(lat_x)
                          for y in np.arange(lat_y) if wall_fn(x, y)])

        d = dict(((k, eval(k)) for k in [
            'lat_x', 'lat_y', 'omega', 't_recordpoints', 'flow_hists', 'walls'
        ]))

        pickle_save(outfile, d)
