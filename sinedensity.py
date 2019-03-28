"""
prescribe an initial density at time t=0 based on a sine wave pattern that varies along the x-axis:

 Simulate the evolution/decay of this disturbance over time.

ARGUMENTS
    --lat_x, --lat_y : int
        overrides default x- and y-dimensions of the entire lattice

    --grid_x, --grid_y : int
        overrides default x- and y-dimensions of the cartesian processor arrangement. If specified, their product must match the number of processors in use.

INPUTS
    None

OUTPUTS
    sinedensity.pkl.gz
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
epsilon = 0.01
timesteps = 5000
rec_interval = 5
omegas = [0.5, 1.0, 1.5]
outfile = 'sinedensity.pkl.gz'

#%% SETUP
t_hist = np.arange(timesteps, step=rec_interval)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up the lattice
lat = Lattice([lat_x, lat_y], grid_dims=grid_dims)

if rank == 0:
    lat.print_info()

# Strength of sine-wave disturbance
sin_x = (epsilon * np.sin(
    2 * np.pi * lat.cell_ranges[0] / lat_x))[:, np.newaxis, np.newaxis]

# add disturbance to current rho
rho_modified = lat.rho() + sin_x

# data structure to store results.
# density and velocity are constant wrt y, so we only need to store one row of x at every timestep
density_hists = {
    omega: np.empty([timesteps // rec_interval, lat_x])
    for omega in omegas
}

#storing x-velocity only
velocity_hists = {
    omega: np.empty([timesteps // rec_interval, lat_x])
    for omega in omegas
}

#%% SIMULATION

for omega in omegas:
    lat.reset_to_eq()
    density_hist = density_hists[omega]
    velocity_hist = velocity_hists[omega]

    # run collision operator once, feeding this prescribed rho in
    lat.collide(rho=rho_modified)

    # begin simulation loop
    for t in range(timesteps):
        lat.halo_copy()
        lat.stream()
        lat.collide(omega=omega)

        if t % rec_interval == 0:
            # save results
            rho_snapshot = lat.gather(lat.rho())
            u_snapshot = lat.gather(lat.u())

            if rank == 0:
                density_hist[t // rec_interval] = rho_snapshot[:, 0, 0]
                velocity_hist[t // rec_interval] = u_snapshot[:, 0, 0]

                # sine wave pattern should be consistent across all y

                assert np.isclose(rho_snapshot, rho_snapshot[:, 0:1]).all()

                assert np.isclose(u_snapshot, u_snapshot[:, 0:1]).all()

#%% SAVE AND EXIT
if rank == 0:

    d = dict(((k, eval(k)) for k in [
        'lat_x', 'lat_y', 'epsilon', 'omegas', 't_hist', 'density_hists',
        'velocity_hists'
    ]))

    pickle_save(outfile, d)
