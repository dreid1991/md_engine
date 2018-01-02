from __future__ import print_function
import sys
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
import pytest
import numpy as np

####################################################################################################
#  DASH
#
#  test_EvaluatorWallHarmonic.py
#
#  Verifies that EvaluatorWallHarmonic returns the correct forces and energies.
#
#  Author: Brian Keene
#
#  Tests:
#
#       test_EvaluatorWallHarmonic_force()
#
#       test_EvaluatorWallHarmonic_force_device()
#
#       test_EvaluatorWallHarmonic_energy()
#
#       test_EvaluatorWallHarmonic_energy_device()
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




def test_EvaluatorWallHarmonic_force(make_state):
    """ analytically verifies that the EvaluatorWallHarmonic returns proper force """
    state = make_state['state']
    springConstant = 5.0
    cutoff = 10.0
    harmonicWall = EvaluatorWallHarmonic(springConstant, cutoff)

    # distancesFromWall; make it so that a couple results not representable as an integer
    distancesFromWall = [1.57079632679, # pi/2
                         3.14159265359, # pi
                         5.0,           # 5
                         6.28318530718, # 2 * pi
                         9.32737905309] # sqrt(87)

    # forceDirection must have a magnitude of 1.0;
    # -- in DASH, this is normalized on instantiation of FixWallHarmonic
    forceDirection = Vector(1.0, 0.0, 0.0)

    # put explicitly the form of the expected force
    def force(distanceFromWall, direction):
        if (distanceFromWall <= cutoff):
            return direction * springConstant * np.abs((cutoff - distanceFromWall))
        else:
            return Vector(0.0, 0.0, 0.0)


    # test for each value of distance from the wall
    tolerance = 1e-10

    for _, item in enumerate(distancesFromWall):
        calculatedForce = harmonicWall.force(item, forceDirection)
        expectedForce   = force(item, forceDirection)
        formatCode = '{:<30s} {:>18.14f} {:>18.14f} {:>18.14f}'
        print(formatCode.format("calculatedForce: ", calculatedForce[0], calculatedForce[1], calculatedForce[2]))
        print(formatCode.format("expectedForce: ",   expectedForce[0], expectedForce[1], expectedForce[2]))

        absoluteDifference = (calculatedForce - expectedForce).len()

        relativeError      = absoluteDifference / (expectedForce.len())
        print("{:<30s} {:>18.14f}".format("percent error: ", relativeError * 100.0))

        assert(relativeError <= tolerance)


def test_EvaluatorWallHarmonic_force_device(make_state):
    """ analytically verifies that the EvaluatorWallHarmonic returns proper force on the GPU """
    state = make_state['state']
    springConstant = 5.0
    cutoff = 10.0
    harmonicWall = EvaluatorWallHarmonic(springConstant, cutoff)

    # distancesFromWall; make it so that a couple results not representable as an integer
    distancesFromWall = [1.57079632679, # pi/2
                         3.14159265359, # pi
                         5.0,           # 5
                         6.28318530718, # 2 * pi
                         9.32737905309] # sqrt(87)

    # forceDirection must have a magnitude of 1.0;
    # -- in DASH, this is normalized on instantiation of FixWallHarmonic
    forceDirection = Vector(1.0, 0.0, 0.0)

    # put explicitly the form of the expected force
    def force(distanceFromWall, direction):
        if (distanceFromWall <= cutoff):
            return direction * springConstant * np.abs((cutoff - distanceFromWall))
        else:
            return Vector(0.0, 0.0, 0.0)


    # test for each value of distance from the wall
    tolerance = 1e-10

    for _, item in enumerate(distancesFromWall):
        calculatedForce = harmonicWall.force_device(item, forceDirection)
        expectedForce   = force(item, forceDirection)
        formatCode = '{:<30s} {:>18.14f} {:>18.14f} {:>18.14f}'
        print(formatCode.format("calculatedForce: ", calculatedForce[0], calculatedForce[1], calculatedForce[2]))
        print(formatCode.format("expectedForce: ",   expectedForce[0], expectedForce[1], expectedForce[2]))

        absoluteDifference = (calculatedForce - expectedForce).len()

        relativeError      = absoluteDifference / (expectedForce.len())
        print("{:<30s} {:>18.14f}".format("percent error: ", relativeError * 100.0))

        assert(relativeError <= tolerance)


def test_EvaluatorWallHarmonic_energy(make_state):
    """ analytically verifies that the EvaluatorWallHarmonic returns proper energy """
    state = make_state['state']
    springConstant = 5.0
    cutoff = 10.0
    harmonicWall = EvaluatorWallHarmonic(springConstant, cutoff)

    distancesFromWall = [1.57079632679, # pi/2
                         3.14159265359, # pi
                         5.0,           # 5
                         6.28318530718, # 2 * pi
                         9.32737905309] # sqrt(87)

    def energy(distanceFromWall):
        if (distanceFromWall < cutoff):
            return 0.5 * springConstant * ((cutoff - distanceFromWall) ** 2.0)
        else:
            return 0.0

    # test for each value of distance from the wall
    tolerance = 1e-10

    for _, item in enumerate(distancesFromWall):
        calculatedEnergy = harmonicWall.energy(item)
        expectedEnergy   = energy(item)
        formatCode = "{:<30s} {:>18.14f}"
        print(formatCode.format("calculatedEnergy: ", calculatedEnergy))
        print(formatCode.format("expectedEnergy: ", expectedEnergy))

        absoluteDifference = np.abs(calculatedEnergy - expectedEnergy)

        relativeError = absoluteDifference / (np.abs(expectedEnergy))
        print(formatCode.format("percentError: ", relativeError * 100.0))

        assert(relativeError <= tolerance)


def test_EvaluatorWallHarmonic_energy_device(make_state):
    """ analytically verifies that the EvaluatorWallHarmonic returns proper energy on the GPU"""
    state = make_state['state']
    springConstant = 5.0
    cutoff = 10.0
    harmonicWall = EvaluatorWallHarmonic(springConstant, cutoff)

    distancesFromWall = [1.57079632679, # pi/2
                         3.14159265359, # pi
                         5.0,           # 5
                         6.28318530718, # 2 * pi
                         9.32737905309] # sqrt(87)

    def energy(distanceFromWall):
        if (distanceFromWall < cutoff):
            return 0.5 * springConstant * ((cutoff - distanceFromWall) ** 2.0)
        else:
            return 0.0

    # test for each value of distance from the wall
    tolerance = 1e-10

    for _, item in enumerate(distancesFromWall):
        calculatedEnergy = harmonicWall.energy_device(item)
        expectedEnergy   = energy(item)
        formatCode = "{:<30s} {:>18.14f}"
        print(formatCode.format("calculatedEnergy: ", calculatedEnergy))
        print(formatCode.format("expectedEnergy: ", expectedEnergy))

        absoluteDifference = np.abs(calculatedEnergy - expectedEnergy)

        relativeError = absoluteDifference / (np.abs(expectedEnergy))
        print(formatCode.format("percentError: ", relativeError * 100.0))

        assert(relativeError <= tolerance)

