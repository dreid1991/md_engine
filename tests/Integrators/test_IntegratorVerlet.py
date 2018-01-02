from __future__ import print_function
import sys
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
import pytest
import numpy as np

####################################################################################################
#  DASH
#
#  IntegratorVerlet.py
#
#  This file considers a series of basic tests with a single particle to verify that the most
#  basic features of IntegratorVerlet are correct.
#
#  Author: Brian Keene
#
#  Tests:
#       test_1_particle_no_vel():
#              a particle with no velocity and no potential has static position
#
#       test_1_particle_const_vel():
#              a particle with constant velocity and no potential ends up in the expected position
#              after some time
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

def test_1_particle_no_vel(make_state):
    '''a particle with no velocity and no potential has static position'''
    # we must add an atom - specify the species (must conform to species added in make_state())
    # and then specify the position
    state = make_state['state']
    initialPosition = Vector(20.,20.,20.)
    species = 'test'
    state.addAtom(species,initialPosition)
    integratorVerlet = IntegratorVerlet(state)

    for a in state.atoms:
        a.vel = Vector(0.0, 0.0, 0.0)

    numTurns = 10000

    # if we run for 10000 turns, with 0 velocity,
    # and then get the final position, it should be identically the same
    integratorVerlet.run(numTurns)

    finalPosition = state.atoms[0].pos
    tol = 1e-10
    # finalPosition and initialPosition are Vector types; use .len() method to get magnitude
    # of their difference, and compare this to our specified tolerance.
    # -- this test will always fail in single precision;
    # -- it should always pass in double precision (deviation is O(10^-12))
    assert( (finalPosition - initialPosition).len() <= tol)

    # End of test #


# for tests that take a long time to run, we only run them if a --runslow flag is passed on
# executing the pytest command
#@pytest.mark.slow
def test_1_particle_const_vel_x(make_state):
    '''a particle with constant velocity and no potential ends up in the expected position
       after some time'''
    state = make_state['state']
    initialPosition = Vector(20.,20.,20.)
    species = 'test'
    state.addAtom(species,initialPosition)
    integratorVerlet = IntegratorVerlet(state)

    velocity = Vector(1.0, 0.0, 0.0)
    for a in state.atoms:
        a.vel = velocity

    numTurns = 10000

    # change in position is simply dt * N * v
    changeInPosition = state.dt * numTurns * velocity[0]

    final_x_position_predicted = initialPosition[0] + changeInPosition
    # if we run for 10000 turns, with 1 velocity in the x direction,
    # then we should be able to trivially predict the final position
    integratorVerlet.run(numTurns)

    final_x_position = state.atoms[0].pos[0]
    final_y_position = state.atoms[0].pos[1]
    final_z_position = state.atoms[0].pos[2]
    tol = 1e-10

    # the x position should match the prediction; the y, z positions should not have changed.
    assert( np.abs(final_x_position - final_x_position_predicted) <= tol)
    assert( np.abs(final_y_position - initialPosition[1]) <= tol)
    assert( np.abs(final_z_position - initialPosition[2]) <= tol)

    # End of test #



