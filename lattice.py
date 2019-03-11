import numpy as np

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

    NC = len(C)

    # distribution probabilities for each channel, in equilibrium
    #TODO rename
    W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

    def __init__(self, x_len, y_len):
        """
        Initialises a lattice with equilibrium conditions
        """

        # we add two to each dimension to allow for the halo
        self.data = np.broadcast_to(self.W[np.newaxis, np.newaxis, :],
                                    (x_len + 2, y_len + 2, self.NC)).copy()

    def core(self):
        """Returns the data from this lattice, excluding the halo cells"""
        return self.data[1:-1, 1:-1, :]

    
    def stream(self, steps=1):
        """roll each of the channels. This uses periodic boundary conditions everywhere."""

        n = np.sum(self.data)

        for i in range(1, self.NC):  #we don't need to do this for channel zero
            # channels move to like channels!
            self.data[:, :, i] = np.roll(
                self.data[:, :, i], self.C[i] * steps, axis=(0, 1))

        # check that particles have been conserved
        assert np.isclose(n, np.sum(self.data))

    def rho(self, core=True):
        """m x n: density at every position"""
        data = self.core() if core else self.data

        rho = np.sum(data, axis=2, keepdims=True)

        return rho

    def j(self, core=True):
        """2 x m x n: density * velocity at each point"""
        data = self.core() if core else self.data

        j = np.einsum('mni,id->mnd', data, self.C)

        return j

    def u(self, core=True, rho=None):
        """m x n x 2: average velocity at each point"""
        #calculate rho
        if rho is None:
            rho = self.rho(
            )  #should have core in here, but will get rid of this
        j = self.j()

        # out= option gives us zeros where the where= condition is not met (i.e. where rho = 0)
        u = np.divide(j, rho, out=np.zeros_like(j), where=(rho != 0))

        return u

    def f_eq(self, u=None):
        """m x n x i: equilibrium flow at each position (given current avg velocity)
        
        optional: prescribed velocity u
        """
        rho = self.rho()
        if u is None: u = self.u(rho=rho)

        cu = np.einsum('id,mnd->mni', self.C, u)

        cu2 = np.power(cu, 2)

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

    def collide(self, omega, prescribed_u=None):
        # prescribed_u (optional) overrides the u calculated from the provided lattice
        diff_to_eq = self.f_eq(prescribed_u) - self.core()
        self.core()[...] = self.core() + (omega * diff_to_eq)