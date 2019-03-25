import numpy as np
from mpi4py import MPI


class Lattice:
    """
    a Lattice

    (Note: north is an INCREASE, so the origin (0,0) is in the lower-left corner)

    Initialises a lattice with equilibrium conditions

    INITIALISATION INPUTS

        lattice_dims: [int x, int y]
            The total size of the lattice being simulated

        grid_dims: [int m, int n], optional
            How to arrange the grid of cells on the available processors
            If this is omitted, an arrangement which minimises the amount of halo copy operations
            will be calculated automatically

        wall_fn: (x, y) -> bool IsWallCell

        drag_fn: (x, y) -> [ux, uy]

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
                 lattice_dims,
                 grid_dims=None,
                 wall_fn=None,
                 drag_fn=None):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.grid_size = self.comm.Get_size()

        self.lattice_dims = lattice_dims

        # if user has not explicitly provided a grid size, then calculate the best one
        if grid_dims is None:
            grid_dims = self.opt_grid_dims(self.grid_size, *lattice_dims)

        assert np.prod(
            grid_dims
        ) == self.grid_size, "Specified grid size does not match the number of processors in use"

        self.grid_dims = grid_dims

        # we want periodicity in all dimensions
        self.cart = self.comm.Create_cart(grid_dims, periods=[True, True])

        # work out the sequence of x-lengths and y-lengths for cells to cover the grid
        self.cell_start_scheme, self.cell_dim_scheme = self.opt_cell_ranges(self.lattice_dims, self.grid_dims)

        coords_list = [self.cart.Get_coords(c) for c in range(self.grid_size)]
        
        self.cell_starts = np.array([(self.cell_start_scheme[0][x], self.cell_start_scheme[1][y]) for x, y in coords_list])
        
        self.cell_dims = np.array([(self.cell_dim_scheme[0][x], self.cell_dim_scheme[1][y]) for x, y in coords_list])
        
        # calculate length in each dimension (will be identical for all cells except the last on each axis)
        self.cell_dims_max = np.max(self.cell_dims, axis=0)

        # fetch values for this particular cell
        cell_start = self.cell_starts[self.rank]
        cell_dim = self.cell_dims[self.rank]

        assert not np.any(
            self.cell_dim_scheme == 0
        ), "Some nodes have nothing to do. Try a larger grid (or fewer nodes)."

        # if self.rank == 0:
            # rank zero will use this map for gathering later
                # self.cell_sizes = 

        # set up array representing the lattice
        # we add two to each dimension to allow for the halo
        self.data = np.broadcast_to(self.W[np.newaxis, np.newaxis, :],
                                    np.append(np.add(cell_dim, 2),
                                              self.NC)).copy()

        # the data from this lattice, excluding the halo cells
        self.core = self.data[1:-1, 1:-1, :]
        assert not self.core.flags.owndata
        assert self.core.flags.writeable

        # we will need these contiguous arrays to receive column data from neighbour cells
        # (rows are already contiguous)
        self.halo_ydec_recvr = np.empty([cell_dim[0] + 2, 1,
                                         self.NC])  #1, not 0, right?
        self.halo_yinc_recvr = np.empty([cell_dim[0] + 2, 1, self.NC])  #TODO

        self.cell_ranges = [
            np.arange(cell_start[d], cell_start[d] + cell_dim[d])
            for d in [0, 1]
        ]

        # work out the locations of dry cells (if any) in this node
        if wall_fn is None:
            self.walls = None
        else:
            self.walls = wall_fn(
                *np.meshgrid(*self.cell_ranges, indexing='ij'))

        # velocity of walls (for the sliding lid thing)
        if drag_fn is None:
            self.drag = None
        else:
            self.drag = drag_fn(*np.meshgrid(*self.cell_ranges, indexing='ij'))

        # these are 2-tuples which each store the rank of the previous (next) lattice on the [x, y] axis
        self.rank_prev, self.rank_next = zip(
            self.cart.Shift(direction=0, disp=1),
            self.cart.Shift(direction=1, disp=1))

    @staticmethod
    def opt_grid_dims(procs, x_len, y_len):
        """
        automatically finds the best grid arrangement, given an available number of processors and a lattice size
        """

        min_ghosts = np.inf
        best_division = None

        # try each possible factorisation [f,g] of the # processors
        for f in range(int(procs**0.5), 0, -1):
            if procs % f == 0:
                g = procs // f

                # try dividing lattice both f*g and g*f
                for m, n in [[f, g], [g, f]]:

                    # actual number will be this * 2
                    ghosts = (m * y_len) + (n * x_len)

                    if ghosts < min_ghosts:
                        min_ghosts = ghosts
                        best_division = [m, n]

        return best_division

    @staticmethod
    def opt_cell_ranges(lat_dims, grid_dims):
        """
        INPUTS
            lat_dim: [int x, int y], np.array shape (2,)
                the size of the entire lattice

            grid_dim: [int m, int n], np.array shape (2,)
                the size of the grid arrangement, where m * n = #processors
        
        OUTPUTS
            cell_starts: np.array (m,n,2,)
                for each cell at location i in m, j in n
                the (x,y) lattice coordinates of its start position
        """

        cell_dims = [None, None]
        cell_starts = [None, None]

        for d in [0, 1]:
        
            int_quotient = lat_dims[d] // grid_dims[d]
            
            num_ints = grid_dims[d] * (int_quotient + 1) - lat_dims[d]

            num_intplusones = grid_dims[d] - num_ints

            cell_dims[d] = [int_quotient + 1] * num_intplusones + [int_quotient] * num_ints
            cell_starts[d] = [sum(cell_dims[d][:i]) for i in range(len(cell_dims[d]))]

        return cell_starts, cell_dims


    def print_info(self):
        print("Simulating {} lattice using a {} process grid".format(
            self.lattice_dims, self.cart.dims))
        print("Cell lengths (x): {}".format(self.cell_dim_scheme[0]))
        print("Cell lengths (y): {}".format(self.cell_dim_scheme[1]))

    def reset_to_eq(self):
        """resets all channel occupation numbers back to equilibrium values"""
        self.data[...] = self.W[np.newaxis,
                                np.newaxis, :]  # broadcast, please.

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
        """
        stream each of the channels in a cell. 
        This uses periodic boundary conditions everywhere.
        """

        n = np.sum(self.data)  #TODO - move out

        # we can start at channel 1, since 0 is the rest channel
        for i in range(1, self.NC):
            # channels move to like channels!
            self.data[:, :, i] = np.roll(
                self.data[:, :, i], self.C[i] * steps, axis=(0, 1))

        if self.walls is not None:
            # bounce channels backward if they are at a dry cell
            self.core[self.walls] = self.core[self.walls][:, self.C_reflection]

            # are walls moving?
            # TODO

        # check that particles have been conserved
        assert np.isclose(n, np.sum(self.data))

    def rho(self):
        """m x n: density at each point"""
        return np.sum(self.core, axis=2, keepdims=True)

    def j(self):
        """m x n x 2: momentum density at each point"""
        return np.einsum('mni,id->mnd', self.core, self.C)

    def u(self, rho=None):
        """m x n x 2: average velocity at each point"""
        if rho is None:
            rho = self.rho()
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

        # for the rest channel, these terms should drop out
        assert cu[0].all() == 0
        assert cu2[0].all() == 0

        return self.W * np.multiply(rho, inside_term)

    def collide(self, omega=1, rho=None, u=None):
        # prescribed_u (optional) overrides the u calculated from the provided lattice
        self.core += omega * (self.f_eq(rho, u) - self.core)

    def gather(self, data):
        depth = data.shape[2]

        # gather all nodes in 1D form array
        telescope = np.empty(
            [self.grid_size,
             np.prod(self.cell_dims_max), depth])

        # pad out any that are undersized (unpredictable results, otherwise)
        self.comm.Gather(
            np.ascontiguousarray(
                np.resize(data, [*self.cell_dims_max, depth])),
            telescope,
            root=0)

        # this will hold the final results
        pool = np.empty([*self.lattice_dims, depth])

        if self.rank == 0:

            for r in range(self.grid_size):

                # cut each cell down to its original size, then reshape into HxW
                tele_input = telescope[r, :np.prod(self.cell_dims[r])].reshape(
                    *self.cell_dims[r], depth)

                # paste it into the final results pool at the appropriate position
                pool[self.cell_starts[r, 0]:self.cell_starts[r, 0] +
                     self.cell_dims[r, 0], self.cell_starts[r, 1]:self.
                     cell_starts[r, 1] + self.cell_dims[r, 1]] = tele_input

        return pool
