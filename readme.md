Implementation of a D2Q9 Lattice-Boltzmann simulator, and various experiments with it. Comprises the code portion of a final submission for the _High Performance Computing with Python_ course.

Our results can be reproduced in full by executing `msub.sh`, and then plotted using `plots.sh`.

All the computation code can be run in parallel with arbitrary lattice sizes, numbers of processors, and processor grid layouts. (Note: plotting code in `plots.sh` and `*_plots.py` should only be run in serial.)

Command line argument for the non-plotting python scripts are:
```
    --lat_x, --lat_y : int
        overrides default x- and y-dimensions of the entire lattice

    --grid_x, --grid_y : int
        overrides default x- and y-dimensions of the cartesian processor arrangement. 
        If specified, their product must match the number of processors in use.
```