#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
# set max execution time
#PBS -l walltime=0:10:00
# code to execute
#PBS -q short_cpuQ

module load mpich-3.2

export OMP_NUM_THREADS=128 # if not using OpenMP, just comment the export out !

mpirun.actual -np 1024  /home/eric.jahnke/exercise1/summa4_mpi 8192 8192 8192

# Compilation on cluster: mpicc -fopenmp -g -Wall summa6.c -o summa6_mpi -lm
# Compilation on MacBook: mpicc -openmp -g -Wall summa6.c -o summa6_mpi -lm