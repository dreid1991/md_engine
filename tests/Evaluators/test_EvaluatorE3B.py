from __future__ import print_function
import sys
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
import pytest
import numpy as np

####################################################################################################
#  DASH
#
#  test_EvaluatorE3B.py
#
#  Verifies that EvaluatorE3B returns the correct forces and energies on host and device.
#  This tests only E3B3.  Note that E3B3 and E3B2 differ only by their prefactors, and
#  have the same functional forms.
#
#  Author: Brian Keene
#
#  Tests:
#
#       test_EvaluatorE3B_force()
#           -TODO
#
#       test_EvaluatorE3B_force_device()
#           -TODO
#
#       test_EvaluatorE3B_energy()
#           -TODO
#
#       test_EvaluatorE3B_energy_device()
#           -TODO
#
####################################################################################################


'''
 see pytest fixtures: here, we create resources that will be used repeatedly;
 note that new instances of the resource are created each time an individual test is run
 as indicated by (scope="function")

'''
@pytest.fixture(scope="function")
def make_state():
    """
        make_state()

        Make a unit of State;
        Set a GPU device for the simulations;
        create a simulation box;
        set the cutoff for potentials;
        set the padding for neighborlisting;
        set the interval for which the neighborlist should be reconstructed;
        set the interval over which the simulation progress should be printed to std::cout;
        set the timestep (dt)
        set the units (real)
        add species O
        add species H
        add species M
    """
    state = State()
    state.deviceManager.setDevice(0)
    state.bounds = Bounds(state, lo = Vector(0, 0, 0), hi = Vector(100.0, 100.0, 100.0))
    state.rCut = 9.0
    state.padding = 1.0
    state.periodicInterval = 1
    state.shoutEvery = 100
    state.dt = 0.001
    state.units.setReal()

    # add H2O
    state.atomParams.addSpecies(handle="OW", mass=15.9994, atomicNum=8)
    state.atomParams.addSpecies(handle="HY", mass=1.008,   atomicNum=1)
    state.atomParams.addSpecies(handle="OM", mass=0.000,   atomicNum=0)
    data = {'state': state}
    return data



def test_EvaluatorE3B_force(make_state):
    """ explicitly verifies that the EvaluatorE3B returns proper force """

    # it is expected that, at the cutoff, the Evaluator returns a force of 0.0;
    # retrieve state
    state = make_state['state']

    # step 1. instantiate the potential
    #E3B= EvaluatorE3B()

    # step 3. take a gromacs E3B trimer and


    # step 4. Define E3B forces for a given r_ij

    # step 5. interface with DASH EvaluatorE3B.force(**args)
    #         and compare with the ones computed here.
    #         --- need to compute both the two body and three body


    # forces not near zero, relative error should be < 1e-10; if absolute value
    # of expected force is < 1e-8, assert that the difference between the two is < 1e-10
    tolerance = 1e-10


    '''
        calculatedForce = .force(sigma, epsilon,cutoff,dr)
        expectedForce   = force(dr)

        formatCode = '{:<30s} {:>18.14f} {:>18.14f} {:>18.14f}'
        print(formatCode.format("dr: ", dr[0], dr[1], dr[2]))
        print(formatCode.format("calculatedForce: ", calculatedForce[0], calculatedForce[1], calculatedForce[2]))
        print(formatCode.format("expectedForce: ",   expectedForce[0], expectedForce[1], expectedForce[2]))
        absoluteDifference = (calculatedForce - expectedForce).len()

        relativeError      = absoluteDifference / (expectedForce.len())
        print("{:<30s} {:>18.14f}\n".format("percent error: ", relativeError * 100.0))

        # so, the 4th entry is designed to have 0 force;
        # this will cause the relativeError to blow up, since expectedForce.len() should be zero;
        # instead, just assert that the absolute difference between these two near zero numbers is also near zero
        if (absoluteDifference > 1e-8):
            assert(relativeError <= tolerance)
        else:
            assert(absoluteDifference <= tolerance)
    '''
    print("Not implemented")

def test_EvaluatorE3B_force_device(make_state):
    """ explicitly verifies that the EvaluatorE3B returns proper force on the GPU """

    #assert(False)
    print("Not implemented")

def test_EvaluatorE3B_energy(make_state):
    """ explicitly verifies that the EvaluatorE3B returns proper energy """
    # it is expected that, at the cutoff, the Evaluator returns a force of 0.0;
    # moreover, the force should be zero at r =
    state = make_state['state']
    print("Not implemented")

def test_EvaluatorE3B_energy_device(make_state):
    """ explicitly verifies that the EvaluatorE3B returns proper energy on the GPU """
    state = make_state['state']
    # sigma, epsilon, cutoff
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0

    print("Not implemented")
    #assert(False)
