#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=out.dat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1

module load boost/1.55+python-2.7-2014q1
module load gcc/4.7
module load cuda/8.0
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/danielreid/md_engine/core/build/src/

python benchmark.py
