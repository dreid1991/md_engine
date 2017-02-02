#module load midway2
module load cuda/7.5
module load cmake

#module load boost/1.55+python-2.7-2014q1
module load boost/1.62.0+openmpi-1.6
#module load gcc
gcc --version
export CC=gcc
export CXX=g++
$LD_LIBRARY_PATH
cmake .. -DPYTHON=1 -DMIDWAY2=1
make -j 4 
