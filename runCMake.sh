module load cmake

module load boost/1.55+python-2.7-2014q1
module load gcc/4.7
module load cuda/8.0
export CC=gcc
export CXX=g++
$LD_LIBRARY_PATH
cmake .. -DPYTHON=1
make -j 4 
