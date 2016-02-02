#!/bin/sh

THIS=$( dirname $0 )
GB=$( cd $THIS ; cd .. ; /bin/pwd )
GB_PY=$GB/python
export PYTHONPATH=$GB_PY:$THIS

swift-t -I $THIS $THIS/pvd-call.swift
