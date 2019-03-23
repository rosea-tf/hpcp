#!/bin/bash -x

module load devel/python/3.5.2
module load mpi/openmpi/3.0-gnu-7.1

mpiexec --bind-to core --map-by core python poiseuille.py
