"""
Simulation of shear wave decay

@author: AlexR
"""
#%% SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpi4py import MPI
from mpl_toolkits.mplot3d import Axes3D

from lattice import Lattice

plt.rcParams.update(plt.rcParamsDefault)

# dimensions of the lattice
# X_FULL_LEN, Y_FULL_LEN = 100, 100
X_FULL_LEN, Y_FULL_LEN = 40, 40

N_TIMESTEPS = 100
CAPTURE_INTERVAL = 10
t_hist = np.arange(N_TIMESTEPS, step=CAPTURE_INTERVAL)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up the lattice
lat = Lattice([X_FULL_LEN, Y_FULL_LEN])

if rank == 0:
    print("Using a {} topology".format(lat.cart.dims))

#%% 3.2: an initial velcoity (u) dist (along y-axis)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def run_shear_decay(lat, omega, epsilon):
    lat.reset_to_eq()

    k = 2 * np.pi / Y_FULL_LEN

    # initial velocity u_x 
    u_x_sin = (epsilon * np.sin(k * lat.cell_ranges[1]))[np.newaxis, :]

    # initial velocity u_y
    u_y_sin = np.zeros_like(u_x_sin)

    u_sin = np.stack([u_x_sin, u_y_sin], axis=-1)

    # run collision operator once, feeding this prescribed u in
    lat.collide(u=u_sin)
    u_initial = lat.gather(lat.u())
    # if rank == 0:
        # assert np.allclose(u_sin, lat.u())
        # assert np.allclose(lat.u(), u_initial[0:30, 0:20])
        # plt.plot(u_initial[0, :, 0])
        # plt.show()


    #find reference position for sin wave peak
    sin_peak_pos_y = np.argmax(u_initial[0, :, 0])

    #storing amplitude of sine wave (i.e. velocity) at each y position
    amplitude_hist = np.empty([N_TIMESTEPS // CAPTURE_INTERVAL, Y_FULL_LEN])

    # SIMULATION
    for t in range(N_TIMESTEPS):

        if t % CAPTURE_INTERVAL == 0:
            u_snapshot = lat.gather(lat.u())
            if rank == 0:
                
                # sine wave pattern should be is consistent across all x
                assert np.isclose(u_snapshot, u_snapshot[0]).all()

                # with this asserted, we can select from only one x position
                amplitude_hist[t // CAPTURE_INTERVAL] = u_snapshot[0, :, 0]

        lat.halo_copy()
        lat.stream()
        lat.collide(omega=omega)

    # CALCULATIONS

    # observed 
    amp_peak_ppn_hist = amplitude_hist[:, sin_peak_pos_y] / amplitude_hist[0, sin_peak_pos_y]

    # observed kinematic viscosity
    viscosity_hist = -(1 / (
            (k**2) * (t_hist + 1))) * np.log(amp_peak_ppn_hist)

    # theoretical viscosity
    viscosity_calc = (1 / 3) * ((1 / omega) - (1 / 2))

    # theoretical sine wive amplitude
    alpha = viscosity_calc * (k**2)
    amp_peak_ppn_calc = np.exp(-alpha * t_hist)

    return u_initial, amplitude_hist, viscosity_hist, viscosity_calc, amp_peak_ppn_hist, amp_peak_ppn_calc

#%%
epsilon=0.01

u_initial_05, amplitude_hist_05, viscosity_hist_05, viscosity_calc_05, amp_peak_ppn_hist_05, amp_peak_ppn_calc_05 = run_shear_decay(lat, omega=0.5, epsilon=epsilon)

u_initial_15, amplitude_hist_15, viscosity_hist_15, viscosity_calc_15, amp_peak_ppn_hist_15, amp_peak_ppn_calc_15 = run_shear_decay(lat, omega=1.5, epsilon=epsilon)

#%%
if rank == 0:
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

    ax[0].set_title('Initial shear wave magnitude')
    ax[0].set_xlabel('$u_x$')
    ax[0].set_ylabel('$y$')
    ax[0].set_yticks([])
    ax[0].plot(u_initial_05[0, :, 0], np.arange(Y_FULL_LEN))

    ax[1].set_title('Initial streaming pattern in lattice')
    ax[1].set_xlabel('$x$')
    ax[1].streamplot(np.arange(X_FULL_LEN), np.arange(Y_FULL_LEN), *np.transpose(u_initial_05, [2,1,0]), linewidth=(3/epsilon)*np.linalg.norm(u_initial_05, axis=2).T)

    plt.savefig('report/figs/shearwave_initial.png', dpi=150, bbox_inches='tight')

    #%%

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].set_title('Amplitude of sine wave over time')
    ax[0].plot(amp_peak_ppn_hist_05, label='Observed ($\omega=0.5$)')
    ax[0].plot(amp_peak_ppn_calc_05, linestyle='-.', label='Predicted ($\omega=0.5$)')
    ax[0].plot(amp_peak_ppn_hist_15, label='Observed ($\omega=1.5$)')
    ax[0].plot(amp_peak_ppn_calc_15, linestyle='-.', label='Predicted ($\omega=1.5$)')
    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('Proportion of initial amplitude')
    # ax[0].legend()

    ax[1].plot(viscosity_hist_05, label='Observed ($\omega=0.5$)')
    ax[1].axhline(viscosity_calc_05, linestyle='-.', color='orange', label="Predicted ($\omega=0.5$)")
    ax[1].plot(viscosity_hist_15, color='green', label='Observed ($\omega=1.5$)')
    ax[1].axhline(viscosity_calc_15, linestyle='-.', color='red', label="Predicted ($\omega=1.5$)")
    ax[1].set_title ('Kinematic Viscosity')
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('Viscosity')
    ax[1].legend()

    plt.savefig('report/figs/shearwave_predictions.png', dpi=150, bbox_inches='tight')

    # %%
    # 3dplot!
    fig = plt.figure(figsize=(10, 4))

    # Make data.
    y = np.arange(Y_FULL_LEN)
    yy, tt = np.meshgrid(y, t_hist)

    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # plot a 3D surface like in the example mplot3d/surface3d_demo
    ax.plot_surface(yy, tt, amplitude_hist_05/epsilon, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)
    ax.set_title('$\omega=0.5$')
    ax.set_xlabel('$y$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('Amplitude')

    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    # plot a 3D surface like in the example mplot3d/surface3d_demo
    ax.plot_surface(yy, tt, amplitude_hist_15/epsilon, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)
    ax.set_title('$\omega=1.5$')
    ax.set_xlabel('$y$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('Amplitude')

    # ax.set_zlim(-0.01, 0.01)
    plt.tight_layout()
    plt.savefig('report/figs/shearwave_time.png', dpi=150, bbox_inches='tight')