#!/bin/sh
set -eu

THIS=$( dirname $0 )
GB=$( cd $THIS ; cd .. ; /bin/pwd )
export PYTHONPATH=$GB/python

# python $GB/miccom-demo/pvd.py
python $GB/miccom-demo/pvd-call.py
