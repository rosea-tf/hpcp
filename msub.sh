#!/bin/bash -x

mkdir output
# clean up previous runs
rm output/*.txt

msub -l nodes=1:ppn=12 sinedensity.sh -o output/sinedensity.txt
msub -l nodes=1:ppn=16 shearwave.sh -o output/shearwave.txt
msub -l nodes=1:ppn=16 poiseuille.sh -o output/poiseuille.txt
msub -l nodes=1:ppn=16 -l walltime=1200 couette.sh -o output/couette.txt

msub -l nodes=1:ppn=1 -l walltime=1200 timer.sh -o output/timer1.txt
msub -l nodes=1:ppn=2 -l walltime=900 timer.sh -o output/timer2.txt
msub -l nodes=1:ppn=4 timer.sh -o output/timer4.txt
msub -l nodes=1:ppn=8 timer.sh -o output/timer.txt
msub -l nodes=1:ppn=16 timer.sh -o output/timer16.txt
msub -l nodes=2:ppn=16 timer.sh -o output/timer32.txt
msub -l nodes=2:ppn=28 timer.sh -o output/timer56.txt
msub -l nodes=4:ppn=28 timer.sh -o output/timer112.txt