#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
# set max execution time
#PBS -l walltime=0:02:00
# code to execute
#PBS -q short_cpuQ
module load mpich-3.2
mpirun.actual -np 16  /home/eric.jahnke/exercise1/summa4_mpi 4 4 4
