#!/bin/bash

#SBATCH --job-name=water
#SBATCH --output=tip4p.dat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=bkeene@uchicago.edu
PATH=$PATH:/project/depablo/bkeene/md_engine/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/depablo/bkeene/md_engine/build
#PATH=$PATH:/home/danielreid/ssages_emre/build7.5/
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/danielreid/ssages_emre/build7.5/danmd

module load cuda/8.0

module load boost/1.62.0+openmpi-1.6

#$LD_LIBRARY_PATH
#mpirun ssages Umbrella.json
python PIMD_TIP4P.py 
#python run.py
#python read_restart.py
#python solvate.py --cbPath=/home/daniel/Documents/poly/lammps_configs/cb
