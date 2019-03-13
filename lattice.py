import numpy as np
from mpi4py import MPI


class Lattice:
    """
    a Lattice

    (Note: north is an INCREASE, so the origin (0,0) is in the lower-left corner)

    """

    # a convenient dictionary allowing us to refer to channels by direction, rather than number
    DIR = {
        'R': 0,
        'E': 1,
        'N': 2,
        'W': 3,
        'S': 4,
        'NE': 5,
        'NW': 6,
        'SW': 7,
        'SE': 8,
    }

    # the velocity vectors [x, y] associated with each of the nine channels
    C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1],
                  [-1, -1], [1, -1]])

    # used for bouncing off wall cells
    C_reflection = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    NC = len(C)

    # distribution probabilities for each channel, in equilibrium
    #TODO rename
    W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

    def __init__(self,
                 x_full_len,
                 y_full_len,
                 x_node_qt=None,
                 y_node_qt=None,
                 wall_fn=None,
                 drag_fn=None):
        """
        Initialises a lattice with equilibrium conditions

        wall_fn: (x, y) -> bool IsWallCell

        """

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        if x_node_qt is None or y_node_qt is None:
            # calculate best fit
            x_node_qt, y_node_qt = self.find_division(self.comm.Get_size(),
                                                      x_full_len, y_full_len)

        # calculate the size of the local node
        assert x_full_len % x_node_qt == 0, "Lattice x-dimension not evenly divisible by number of nodes"
        assert y_full_len % y_node_qt == 0, "Lattice y-dimension not evenly divisible by number of nodes"

        self.x_full_len, self.y_full_len = x_full_len, y_full_len

        x_len, y_len = x_full_len // x_node_qt, y_full_len // y_node_qt

        # we add two to each dimension to allow for the halo
        self.data = np.broadcast_to(self.W[np.newaxis, np.newaxis, :],
                                    (x_len + 2, y_len + 2, self.NC)).copy()

        # the data from this lattice, excluding the halo cells
        self.core = self.data[1:-1, 1:-1, :]
        assert not self.core.flags.owndata
        assert self.core.flags.writeable

        # we will need these contiguous arrays to receive column data from neighbour cells
        # (rows are already contiguous)
        self.halo_ydec_recvr = np.empty([x_len + 2, 1, self.NC])
        self.halo_yinc_recvr = np.empty([x_len + 2, 1, self.NC])

        # we want periodicity in all dimensions - for now
        # the next line will throw an exception if x_node_qt * y_node_qt != comm.Get_size()
        self.cart = self.comm.Create_cart([x_node_qt, y_node_qt],
                                          periods=[True, True])

        # calculate range of x and y in the full lattice represented by this node
        x_coord, y_coord = self.cart.coords
        self.x_range = np.arange(x_coord * x_len, (x_coord + 1) * x_len)
        self.y_range = np.arange(y_coord * y_len, (y_coord + 1) * y_len)

        # work out the locations of dry cells (if any) in this node
        if wall_fn is None:
            self.walls = None
        else:
            self.walls = wall_fn(
                *np.meshgrid(self.x_range, self.y_range, indexing='ij'))

        # velocity of walls (for the sliding lid thing)
        if drag_fn is None:
            self.drag = None
        else:
            self.drag = drag_fn(
                *np.meshgrid(self.x_range, self.y_range, indexing='ij'))

        # these are 2-tuples which each store the rank of the previous (next) lattice on the [x, y] axis
        self.rank_prev, self.rank_next = zip(
            self.cart.Shift(direction=0, disp=1),
            self.cart.Shift(direction=1, disp=1))

    @staticmethod
    def find_division(n, x_len, y_len):
        for f in range(int(n**0.5), 0, -1):
            if n % f == 0:
                g = n // f

                if x_len % f == 0 and y_len % g == 0:
                    return [f, g]
                if x_len % g == 0 and y_len % f == 0:
                    return [g, f]

        raise Exception(
            "Could not arrange ({}, {}) lattice on {} processors".format(
                x_len, y_len, n))

    def reset_to_eq(self):
        self.data[...] = self.W[np.newaxis, np.newaxis, :]  #broadcast, please.

    def halo_copy(self):
        """copies data into the halo cells of all lattices"""

        # Send to next x, recv from prev x
        self.comm.Sendrecv(
            self.data[-2:-1, :],
            self.rank_next[0],
            recvbuf=self.data[0:1, :],
            source=self.rank_prev[0])

        # send to prev x, recv from next x
        self.comm.Sendrecv(
            self.data[1:2, :],
            self.rank_prev[0],
            recvbuf=self.data[-1:],
            source=self.rank_next[0])

        # Send to next y, recv from prev y
        self.comm.Sendrecv(
            np.ascontiguousarray(self.data[:, -2:-1]),
            self.rank_next[1],
            recvbuf=self.halo_ydec_recvr,
            source=self.rank_prev[1])

        # send to prev y, recv from next y
        self.comm.Sendrecv(
            np.ascontiguousarray(self.data[:, 1:2]),
            self.rank_prev[1],
            recvbuf=self.halo_yinc_recvr,
            source=self.rank_next[1])

        # copy contiguous temporary buffers into non-contiguous halo columns
        self.data[:, 0:1] = self.halo_ydec_recvr
        self.data[:, -1:] = self.halo_yinc_recvr

    def stream(self, steps=1):
        """roll each of the channels. This uses periodic boundary conditions everywhere."""

        n = np.sum(self.data)

        for i in range(1, self.NC):  #we don't need to do this for channel zero
            # channels move to like channels!
            self.data[:, :, i] = np.roll(
                self.data[:, :, i], self.C[i] * steps, axis=(0, 1))

        if self.walls is not None:
            # bounce channels backward
            self.core[self.walls] = self.core[self.walls][:, self.C_reflection]

            # are walls moving?
            # TODO

        # check that particles have been conserved
        assert np.isclose(n, np.sum(self.data))

    def rho(self):
        """m x n: density at every position"""
        rho = np.sum(self.core, axis=2, keepdims=True)

        return rho

    def j(self):
        """2 x m x n: density * velocity at each point"""
        j = np.einsum('mni,id->mnd', self.core, self.C)

        return j

    def u(self, core=True, rho=None):
        """m x n x 2: average velocity at each point"""
        #calculate rho
        if rho is None:
            rho = self.rho(
            )  #should have core in here, but will get rid of this
        j = self.j()

        # out= option gives us zeros where the where= condition is not met (i.e. where rho = 0)
        return np.divide(j, rho, out=np.zeros_like(j), where=(rho != 0))

    def f_eq(self, rho=None, u=None):
        """m x n x i: equilibrium flow at each position (given current avg velocity)
        
        optional: prescribed velocity u
        """
        if rho is None: rho = self.rho()
        if u is None: u = self.u(rho=rho)

        cu = np.einsum('id,mnd->mni', self.C, u)

        cu2 = cu**2

        u2 = np.sum(np.power(u, 2), axis=2, keepdims=True)

        inside_term = 1 + (3 * cu) + (9 / 2) * cu2 - (3 / 2) * u2
        # print ('cu', cu.shape)
        # print ('cu2', cu2.shape)
        # print ('u', u.shape)
        # print ('u2', u2.shape)
        # print ('it', inside_term.shape)
        # for the rest channel, these terms should drop out
        assert cu[0].all() == 0
        assert cu2[0].all() == 0

        return self.W * np.multiply(rho, inside_term)

    def collide(self, omega=1, rho=None, u=None):
        # prescribed_u (optional) overrides the u calculated from the provided lattice
        # TODO - make this a function?
        self.core += omega * (self.f_eq(rho, u) - self.core)

    def gather(self, data):

        # pool = np.empty([self.x_full_len, self.y_full_len] + list(data.shape[2:]))

        telescope = np.empty([self.comm.Get_size()] + list(data.shape))

        self.comm.Gather(np.ascontiguousarray(data), telescope, root=0)

        data_x_len, data_y_len = data.shape[0:2]

        pool = np.empty([
            data_x_len * self.cart.dims[0], data_y_len * self.cart.dims[1],
            *data.shape[2:]
        ])

        if self.rank == 0:
            # is there an easier way to do this?
            for r in range(telescope.shape[0]):
                rc = self.cart.Get_coords(r)
                pool[rc[0] * data_x_len:(rc[0] + 1) * data_x_len, rc[1] *
                     data_y_len:(rc[1] + 1) * data_y_len] = telescope[r]

        return pool