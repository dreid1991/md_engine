from __future__ import print_function
import sys
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
import pytest
import numpy as np

####################################################################################################
#  DASH
#
#  test_EvaluatorWallLJ126.py
#
#  Verifies that EvaluatorWallLJ126 returns the correct forces and energies on host and device.
#
#  Author: Brian Keene
#
#  Tests:
#
#       test_EvaluatorWallLJ126_force()
#
#       test_EvaluatorWallLJ126_force_device()
#
#       test_EvaluatorWallLJ126_energy()
#
#       test_EvaluatorWallLJ126_energy_device()
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
        set the units (LJ)
        add a species 'test' with mass 1 and atomicNum 1
    """
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



def test_EvaluatorWallLJ126_force(make_state):
    """ analytically verifies that the EvaluatorWallLJ126 returns proper force """

    state = make_state['state']
    # sigma, epsilon, cutoff
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0
    LJWall = EvaluatorWallLJ126(sigma, epsilon, cutoff)

    # distancesFromWall; make it so that a couple results not representable as an integer
    distancesFromWall = [1.57079632679, # pi/2
                         3.14159265359, # pi
                         5.0,           # 5
                         6.28318530718, # 2 * pi
                         9.32737905309]

    # forceDirection must have a magnitude of 1.0;
    # -- in DASH, this is normalized on instantiation of FixWallHarmonic
    forceDirection = Vector(1.0, 0.0, 0.0)

    # put explicitly the form of the expected force
    # -- for this potential, the energy is shifted, but not the force
    def force(distanceFromWall, direction):
        if (distanceFromWall < cutoff):
            # -d/dr(U(r)):
            sig6OverR6 = (( sigma * sigma ) ** 3.0) / ( (distanceFromWall * distanceFromWall) ** 3.0)
            sig12OverR12 = sig6OverR6 * sig6OverR6

            forceScalar = (24.0 * epsilon / distanceFromWall) * ( (2.0 * sig12OverR12) - sig6OverR6)
            return (direction * forceScalar)
        else:
            return Vector(0.0, 0.0, 0.0)


    # test for each value of distance from the wall
    tolerance = 1e-10

    for _, item in enumerate(distancesFromWall):
        calculatedForce = LJWall.force(item, forceDirection)
        expectedForce   = force(item, forceDirection)

        formatCode = '{:<30s} {:>18.14f} {:>18.14f} {:>18.14f}'
        print(formatCode.format("calculatedForce: ", calculatedForce[0], calculatedForce[1], calculatedForce[2]))
        print(formatCode.format("expectedForce: ",   expectedForce[0], expectedForce[1], expectedForce[2]))

        absoluteDifference = (calculatedForce - expectedForce).len()

        relativeError      = absoluteDifference / (expectedForce.len())
        print("{:<30s} {:>18.14f}".format("percent error: ", relativeError * 100.0))

        assert(relativeError <= tolerance)


def test_EvaluatorWallLJ126_force_device(make_state):
    """ analytically verifies that the EvaluatorWallLJ126 returns proper force on the GPU """

    state = make_state['state']
    # sigma, epsilon, cutoff
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0
    LJWall = EvaluatorWallLJ126(sigma, epsilon, cutoff)

    # distancesFromWall; make it so that a couple results not representable as an integer
    distancesFromWall = [1.57079632679, # pi/2
                         3.14159265359, # pi
                         5.0,           # 5
                         6.28318530718, # 2 * pi
                         9.32737905309]

    # forceDirection must have a magnitude of 1.0;
    # -- in DASH, this is normalized on instantiation of FixWallHarmonic
    forceDirection = Vector(1.0, 0.0, 0.0)

    # put explicitly the form of the expected force
    # -- for this potential, the energy is shifted, but not the force
    def force(distanceFromWall, direction):
        if (distanceFromWall < cutoff):
            # -d/dr(U(r)):
            sig6OverR6 = (( sigma * sigma ) ** 3.0) / ( (distanceFromWall * distanceFromWall) ** 3.0)
            sig12OverR12 = sig6OverR6 * sig6OverR6

            forceScalar = (24.0 * epsilon / distanceFromWall) * ( (2.0 * sig12OverR12) - sig6OverR6)
            return (direction * forceScalar)
        else:
            return Vector(0.0, 0.0, 0.0)


    # test for each value of distance from the wall
    tolerance = 1e-10

    for _, item in enumerate(distancesFromWall):
        calculatedForce = LJWall.force_device(item, forceDirection)
        expectedForce   = force(item, forceDirection)

        formatCode = '{:<30s} {:>18.14f} {:>18.14f} {:>18.14f}'
        print(formatCode.format("calculatedForce: ", calculatedForce[0], calculatedForce[1], calculatedForce[2]))
        print(formatCode.format("expectedForce: ",   expectedForce[0], expectedForce[1], expectedForce[2]))

        absoluteDifference = (calculatedForce - expectedForce).len()

        relativeError      = absoluteDifference / (expectedForce.len())
        print("{:<30s} {:>18.14f}".format("percent error: ", relativeError * 100.0))

        assert(relativeError <= tolerance)


def test_EvaluatorWallLJ126_energy(make_state):
    """ analytically verifies that the EvaluatorWallLJ126 returns proper energy """
    state = make_state['state']
    # sigma, epsilon, cutoff
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0

    sigOverCut = sigma / cutoff
    sig6OverCut6 = sigOverCut ** 6.0
    sig12OverCut12 = sig6OverCut6 * sig6OverCut6

    engShift = -4.0 * epsilon * (sig12OverCut12 - sig6OverCut6)

    LJWall = EvaluatorWallLJ126(sigma, epsilon, cutoff)

    # distancesFromWall; make it so that a couple results not representable as an integer
    distancesFromWall = [1.57079632679, # pi/2
                         3.14159265359, # pi
                         5.0,           # 5
                         6.28318530718, # 2 * pi
                         9.32737905309,
                         0.9999999*cutoff]

    # forceDirection must have a magnitude of 1.0;
    # -- in DASH, this is normalized on instantiation of FixWallHarmonic
    forceDirection = Vector(1.0, 0.0, 0.0)


    # put explicitly the form of the expected force
    # -- for this potential, the energy is shifted, but not the force
    def energy(distanceFromWall):
        if (distanceFromWall < cutoff):
            # -d/dr(U(r)):
            sig6OverR6 = (( sigma * sigma ) ** 3.0) / ( (distanceFromWall * distanceFromWall) ** 3.0)
            sig12OverR12 = sig6OverR6 * sig6OverR6
            result = (4.0 * epsilon ) * ( sig12OverR12 - sig6OverR6) + engShift
            return result
        else:
            return 0.0


    # test for each value of distance from the wall
    tolerance = 1e-10

    for _, item in enumerate(distancesFromWall):
        calculatedEnergy = LJWall.energy(item)
        expectedEnergy   =  energy(item)

        formatCode = "{:<30s} {:>18.14f}"
        print(formatCode.format("calculatedEnergy: ", calculatedEnergy))
        print(formatCode.format("expectedEnergy: ", expectedEnergy))

        absoluteDifference = np.abs(calculatedEnergy - expectedEnergy)
        magnitude = np.abs(expectedEnergy)
        relativeError = absoluteDifference / (np.abs(expectedEnergy))
        print(formatCode.format("percentError: ", relativeError * 100.0))
        # even in double, some numerical error occurs!
        if (magnitude > 1e-5):
            tolerance = 1e-10
        else:
            tolerance = 1e-8


        assert(relativeError <= tolerance)




def test_EvaluatorWallLJ126_energy_device(make_state):
    """ analytically verifies that the EvaluatorWallLJ126 returns proper energy on the GPU """
    state = make_state['state']
    # sigma, epsilon, cutoff
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0

    sigOverCut = sigma / cutoff
    sig6OverCut6 = sigOverCut ** 6.0
    sig12OverCut12 = sig6OverCut6 * sig6OverCut6

    engShift = -4.0 * epsilon * (sig12OverCut12 - sig6OverCut6)

    LJWall = EvaluatorWallLJ126(sigma, epsilon, cutoff)

    # distancesFromWall; make it so that a couple results not representable as an integer
    distancesFromWall = [1.57079632679, # pi/2
                         3.14159265359, # pi
                         5.0,           # 5
                         6.28318530718, # 2 * pi
                         9.32737905309,
                         0.9999999*cutoff]

    # forceDirection must have a magnitude of 1.0;
    # -- in DASH, this is normalized on instantiation of FixWallHarmonic
    forceDirection = Vector(1.0, 0.0, 0.0)


    # put explicitly the form of the expected force
    # -- for this potential, the energy is shifted, but not the force
    def energy(distanceFromWall):
        if (distanceFromWall < cutoff):
            # -d/dr(U(r)):
            sig6OverR6 = (( sigma * sigma ) ** 3.0) / ( (distanceFromWall * distanceFromWall) ** 3.0)
            sig12OverR12 = sig6OverR6 * sig6OverR6
            result = (4.0 * epsilon ) * ( sig12OverR12 - sig6OverR6) + engShift
            return result
        else:
            return 0.0


    # test for each value of distance from the wall
    tolerance = 1e-10

    for _, item in enumerate(distancesFromWall):
        calculatedEnergy = LJWall.energy_device(item)
        expectedEnergy   =  energy(item)

        formatCode = "{:<30s} {:>18.14f}"
        print(formatCode.format("calculatedEnergy: ", calculatedEnergy))
        print(formatCode.format("expectedEnergy: ", expectedEnergy))

        absoluteDifference = np.abs(calculatedEnergy - expectedEnergy)
        magnitude = np.abs(expectedEnergy)
        relativeError = absoluteDifference / (np.abs(expectedEnergy))
        print(formatCode.format("percentError: ", relativeError * 100.0))
        # even in double, some numerical error occurs!
        if (magnitude > 1e-5):
            tolerance = 1e-10
        else:
            tolerance = 1e-8


        assert(relativeError <= tolerance)
