"""
Simulation of shear wave decay

@author: AlexR
"""
#%% SETUP

import numpy as np
from mpi4py import MPI
import _pickle as pickle
import argparse
from lattice import Lattice
import os
import gzip

#%%
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

#%% SET PARAMETERS
lat_x = 400
lat_y = 300
epsilon = 0.01
timesteps = 1000 #TODO
rec_interval = 10
omegas = [0.5, 0.75, 1.0, 1.25, 1.5]
outfile = 'shearwave.pkl.gz'

t_hist = np.arange(timesteps, step=rec_interval)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    grid_dims = [
        args.grid_x, args.grid_y
    ] if args.grid_x is not None and args.grid_y is not None else None
except:
    grid_dims = None

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

del u_x_sin, u_y_sin

# run collision operator once, feeding this prescribed u in
lat.collide(u=u_sin)
u_initial = lat.gather(lat.u())

#storing amplitude of sine wave (i.e. velocity) at each y position
amplitude_hists = {omega: np.empty(
    [timesteps // rec_interval, lat_y]) for omega in omegas}  #r0

# density_hist = {omega: np.empty(
    # [timesteps // rec_interval, lat_y]) for omega in omegas}

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

#%% CALCULATIONS

if rank == 0:

    # save variables and exit
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    d = dict(((v, eval(v)) for v in [
        'lat_x', 'lat_y', 'omegas', 'epsilon', 'u_initial', 't_hist', 'k', 'amplitude_hists'
    ]))


    outpath = os.path.join(pickle_path, outfile)

    pickle.dump(d, gzip.open(outpath, 'wb'))

    print("Results saved to " + outpath)


#%%
