#!/bin/bash -x

module load devel/python/3.5.2

python sinedensity_plots.py
python shearwave_plots.py
python poiseuille_plots.py
python couette_plots.py
python timer_plots.py