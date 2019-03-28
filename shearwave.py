"""
 Prescribe an initial velocity at time t= 0 based on a shearwave pattern that varies along the y-axis.
 Simulate the decay of this shearwave over time.

ARGUMENTS
    --lat_x, --lat_y : int
        overrides default x- and y-dimensions of the entire lattice

    --grid_x, --grid_y : int
        overrides default x- and y-dimensions of the cartesian processor arrangement. If specified, their product must match the number of processors in use.

INPUTS
    None

OUTPUTS
    shearwave.pkl.gz
        Compressed results file. Used for plotting.

@author: AlexR
"""

#%% SETUP

import numpy as np
from mpi4py import MPI

import _pickle as pickle
from lattice import Lattice
from utils import fetch_dim_args, pickle_save

#%% SET PARAMETERS

[lat_x, lat_y], grid_dims = fetch_dim_args(lat_default=[400, 300])
epsilon = 0.01
timesteps = 10000
rec_interval = 100
omegas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
outfile = 'shearwave.pkl.gz'

#%% SETUP

t_hist = np.arange(timesteps, step=rec_interval)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up the lattice
lat = Lattice([lat_x, lat_y], grid_dims=grid_dims)

if rank == 0:
    lat.print_info()

lat.reset_to_eq()

k = 2 * np.pi / lat_y

# initial velocity u_x
u_x_sin = (epsilon * np.sin(k * lat.cell_ranges[1]))[np.newaxis, :]

# initial velocity u_y
u_y_sin = np.zeros_like(u_x_sin)

u_sin = np.stack([u_x_sin, u_y_sin], axis=-1)

# run collision operator once, feeding this prescribed u in
lat.collide(u=u_sin)
u_initial = lat.gather(lat.u())

#storing amplitude of sine wave (i.e. velocity) at each y position
amplitude_hists = {
    omega: np.empty([timesteps // rec_interval, lat_y])
    for omega in omegas
}  #r0

#%% SIMULATION
for omega in omegas:
    lat.reset_to_eq()
    # run collision operator once, feeding this prescribed u in
    lat.collide(u=u_sin)

    # density_hist = density_hists[omega]
    amplitude_hist = amplitude_hists[omega]

    for t in range(timesteps):

        if t % rec_interval == 0:
            # rho_snapshot = lat.gather(lat.rho())
            u_snapshot = lat.gather(lat.u())

            if rank == 0:

                # sine wave pattern should be consistent across all x
                assert np.isclose(u_snapshot, u_snapshot[0]).all()

                # with this asserted, we can select from only one x position
                amplitude_hist[t // rec_interval] = u_snapshot[0, :, 0]
                # density_hist[t // rec_interval] = rho_snapshot[0, :, 0]

        lat.halo_copy()
        lat.stream()
        lat.collide(omega=omega)

#%% SAVE RESULTS

if rank == 0:

    d = dict(((v, eval(v)) for v in [
        'lat_x', 'lat_y', 'omegas', 'epsilon', 'u_initial', 't_hist', 'k',
        'amplitude_hists'
    ]))

    pickle_save(outfile, d)
