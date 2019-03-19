"""
Simulation of shear wave decay

@author: AlexR
"""
# SETUP

import numpy as np
from mpi4py import MPI
import _pickle as pickle
import argparse
from lattice import Lattice
import os

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

t_hist = np.arange(args.timesteps, step=args.interval)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

grid_dims = [args.grid_x, args.grid_y
             ] if args.grid_x is not None and args.grid_y is not None else None

# set up the lattice
lat = Lattice([args.lat_x, args.lat_y], grid_dims=grid_dims)

if rank == 0:
    lat.print_info()

omega = args.omega
epsilon = args.epsilon

lat.reset_to_eq()

k = 2 * np.pi / lat.lattice_dims[1]

# initial velocity u_x
u_x_sin = (epsilon * np.sin(k * lat.cell_ranges[1]))[np.newaxis, :]

# initial velocity u_y
u_y_sin = np.zeros_like(u_x_sin)

u_sin = np.stack([u_x_sin, u_y_sin], axis=-1)

del u_x_sin, u_y_sin

# run collision operator once, feeding this prescribed u in
lat.collide(u=u_sin)
u_initial = lat.gather(lat.u())

#find reference position for sin wave peak
sin_peak_pos_y = np.argmax(u_initial[0, :, 0])  #r0

#storing amplitude of sine wave (i.e. velocity) at each y position
amplitude_hist = np.empty(
    [args.timesteps // args.interval, lat.lattice_dims[1]])  #r0

# SIMULATION
for t in range(args.timesteps):

    if t % args.interval == 0:
        u_snapshot = lat.gather(lat.u())
        if rank == 0:

            # sine wave pattern should be is consistent across all x
            assert np.isclose(u_snapshot, u_snapshot[0]).all()

            # with this asserted, we can select from only one x position
            amplitude_hist[t // args.interval] = u_snapshot[0, :, 0]

    lat.halo_copy()
    lat.stream()
    lat.collide(omega=omega)

# CALCULATIONS

if rank == 0:
    # observed sine wave amplitude
    amp_peak_ppn_hist = amplitude_hist[:, sin_peak_pos_y] / amplitude_hist[
        0, sin_peak_pos_y]

    # observed kinematic viscosity
    viscosity_hist = -(1 / ((k**2) * (t_hist + 1))) * np.log(amp_peak_ppn_hist)

    # theoretical viscosity
    viscosity_calc = (1 / 3) * ((1 / omega) - (1 / 2))

    # theoretical sine wive amplitude
    alpha = viscosity_calc * (k**2)
    amp_peak_ppn_calc = np.exp(-alpha * t_hist)

    # save variables and exit
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    d = dict(((k, eval(k)) for k in [
        'omega', 'epsilon', 'u_initial', 't_hist', 'amplitude_hist',
        'viscosity_hist', 'viscosity_calc', 'amp_peak_ppn_hist',
        'amp_peak_ppn_calc'
    ]))

    pickle.dump(d, open(os.path.join(pickle_path, args.output), 'wb'))

    print("Results saved to ./pickles/{}".format(args.output))