#!/bin/bash -x

module load devel/python/3.5.2
module load mpi/openmpi/3.0-gnu-7.1

# run for each of three methods we're looking at
for i in 0 1 2
do
echo METHOD i
mpiexec --bind-to core --map-by core python timer.py $i
done