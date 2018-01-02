from __future__ import print_function
import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
import pytest
import numpy as np

####################################################################################################
#  DASH
#
#  test_1_particle.py
#
#  This file considers a series of basic tests with a single particle to verify that the most
#  basic features are correct.  It is to be run for any update to the master branch.
#  It also verifies that force and energies from wall potentials and external fields are correct.
#
#  Author: Brian Keene
#
#  Tests:
#
#       test_1_particle_potential_1():
#              a particle initialized in some potential well with kinetic energy less than
#              the potential well depth stays in that well
#
#       test_1_particle_potential_2():
#              a particle initialized in some potential well with kinetic energy greater than
#              the potential well depth exits that well; given knowledge of the depth of the well,
#              we can assert the remaining kinetic energy in the absence of the potential
#
#       test_1_particle_fixExternalHarmonic():
#
#       test_1_particle_EvaluatorExternalHarmonic():
#
#       test_1_particle_fixExternalQuartic():
#
#       test_1_particle_EvaluatorExternalQuartic():
#
#       test_1_particle_fixWallLJ126():
#
#       test_1_particle_EvaluatorWallLJ126():
#              analytically verifies that the EvaluatorWallLJ126 returns proper force and energy
#
#       test_1_particle_fixWallHarmonic():
#
####################################################################################################

'''
 see pytest fixtures: here, we create resources that will be used repeatedly;
 note that new instances of the resource are created each time an individual test is run
 as indicated by (scope="function")

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

    data = {'state': state}
    return data

def test_1_particle_potential_1(make_state):
    '''a particle initialized in some potential well with initial kinetic energy less than
       the potential well depth stays in that well '''
    state = make_state['state']
    initialPosition = Vector(3.,20.,20.)
    species = 'test'
    state.addAtom(species,initialPosition)
    integratorVerlet = IntegratorVerlet(state)

    # add a LJ Wall Potential;
    # initialize the particle with 0 KE;
    # place it somewhere within the potential well
    # expect that after some 100k steps, the particle is
    # still within the bounds of the potential.

    assert(True)

def test_1_particle_potential_2(make_state):
    '''a particle initialized in some potential well with kinetic energy greater than
       the potential well depth exits that well; given knowledge of the depth of the well,
       we can assert the remaining kinetic energy in the absence of the potential'''
    state = make_state['state']
    assert(True)


