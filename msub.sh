#!/bin/bash -x

msub -l nodes=1:ppn=12 sinedensity.sh -o sinedensity
msub -l nodes=1:ppn=16 shearwave.sh -o shearwave
msub -l nodes=1:ppn=16 poiseuille.sh -o poiseuille
msub -l nodes=1:ppn=16 -l walltime=1200 couette.sh -o couette

msub -l nodes=1:ppn=1 -l walltime=1200 timer.sh -o timer
msub -l nodes=1:ppn=2 -l walltime=900 timer.sh -o timer
msub -l nodes=1:ppn=4 timer.sh -o timer
msub -l nodes=1:ppn=8 timer.sh -o timer
msub -l nodes=1:ppn=16 timer.sh -o timer
msub -l nodes=2:ppn=16 timer.sh -o timer
msub -l nodes=2:ppn=28 timer.sh -o timer56.out
msub -l nodes=4:ppn=28 timer.sh -o timer