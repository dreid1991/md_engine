from __future__ import print_function
import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
import pytest
import numpy as np



####################################################################################################
#  DASH
#
#  test_2_particle.py
#
#  This file considers a series of basic tests for fixes that can be modelled with a two-particle
#  system.
#
#  Author: Brian Keene
#
#  Tests:
#        - test_2_particles_fix2d()
#        - test_2_particles_fixBondFENE()
#        - test_2_particles_fixBondHarmonic()
#        - test_2_particles_fixBondQuartic()
#        - test_2_particles_fixChargeEwald()
#        - test_2_particles_fixChargePairDSF()
#        - test_2_particles_
#
#
#
#
####################################################################################################

'''
 see pytest fixtures: here, we create resources that will be used repeatedly;
 note that new instances of the resource are created each time an individual test is run
 as indicated by (scope="function")

 So, as it as, this is not being used as a pytest fixture;
 to do so, we would want to have the functions that need these resources to inherit from
 the make_state() function;

 but, this is easier
'''
@pytest.fixture(scope="function")
def make_state():
    '''
        make_state()

        Make a unit of State;
        Set a GPU device for the simulations;
        create a simulation box;
        set the cutoff for potentials;
        set the padding for neighborlisting;
        set the interval for which the neighborlist should be reconstructed;
        set the interval over which the simulation progress should be printed to std::cout;
        set the timestep (dt)
        set the units (LJ)
        add a species 'test' with mass 1 and atomicNum 1
    '''
    state = State()
    state.deviceManager.setDevice(0)
    state.bounds = Bounds(state, lo = Vector(0, 0, 0), hi = Vector(100.0, 100.0, 100.0))
    state.rCut = 9.0
    state.padding = 1.0
    state.periodicInterval = 1
    state.shoutEvery = 100
    state.dt = 0.005
    state.units.setLJ()
    state.atomParams.addSpecies(handle='test', mass=1, atomicNum=1)
    return state

