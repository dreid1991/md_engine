from __future__ import print_function
import sys
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
import pytest
import numpy as np

####################################################################################################
#  DASH
#
#  test_EvaluatorLJ.py
#
#  Verifies that EvaluatorLJ returns the correct forces and energies on host and device.
#
#  Author: Brian Keene
#
#  Tests:
#
#       test_EvaluatorLJ_force()
#           -last edited: BK, 2 January 2018
#
#       test_EvaluatorLJ_force_device()
#           -TODO
#
#       test_EvaluatorLJ_energy()
#           -TODO
#
#       test_EvaluatorLJ_energy_device()
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



def test_EvaluatorLJ_force(make_state):
    """ analytically verifies that the EvaluatorLJ returns proper force """

    state = make_state['state']
    # sigma, epsilon, cutoff; no significance to these values, just pick a number
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0
    LJPotential = EvaluatorLJ()

    sig2 = sigma * sigma
    sig6 = sig2 ** 3.0
    sig12 = sig6 * sig6

    # as currently implemented in PairEvaluateIso, dr = bounds.minImage(pos - otherPos),
    # where we are computing the force on atom with position 'pos'

    # put particle A here, then paticle B, both as vector
    positions_A = [Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0)]

    positions_B = [Vector(9.0, 10.0, 10.0),
                   Vector(8.0, 10.0, 10.0),
                   Vector(7.0, 10.0, 10.0),
                   Vector(10.0 - (sigma * (2.0 ** (1.0 / 6.0))), 10.0, 10.0),
                   Vector(9.23472, 9.84752, 8.293478),
                   Vector(11.0, 9.0, 12.0)]

    # use the same minImage function; make an instance of BoundsGPU
    boundsGPU = BoundsGPU(state.bounds.lo, state.bounds.hi - state.bounds.lo, Vector(1,1,1))

    # put explicitly the form of the expected force
    def force(dr):
        rij_scalar = dr.len()
        if (rij_scalar < cutoff):
            # -d/dr(U(r)):
            # take out a factor of r^-7; direction should be \hat r_{ij}, but we just have vector r_{ij},
            # i.e. it is not normalized - so, multiply by another factor of r^-1;
            r2inv = 1.0 / (rij_scalar ** 2.0)
            r6inv = r2inv * r2inv * r2inv
            forceScalar = r6inv * r2inv * 24.0 * epsilon * ( (2.0 * sig12 * r6inv) - (sig6))
            return (dr * forceScalar)
        else:
            return Vector(0.0, 0.0, 0.0)


    # test for each value of distance from the wall
    tolerance = 1e-10

    for index, _ in enumerate(positions_A):
        pos      = positions_A[index]
        otherPos = positions_B[index]
        toMinImage = pos - otherPos
        dr = boundsGPU.minImage(toMinImage) # exact syntax as in PairEvaluateIso

        # DASH force
        calculatedForce = LJPotential.force(sigma, epsilon,cutoff,dr)
        # test force
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


def test_EvaluatorLJ_force_device(make_state):
    """ analytically verifies that the EvaluatorLJ returns proper force on the GPU """
    print("test_EvaluatorLJ_force_device is not yet implemented")

def test_EvaluatorLJ_energy(make_state):
    """ analytically verifies that the EvaluatorLJ returns proper energy """
    state = make_state['state']
    # sigma, epsilon, cutoff
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0

    sigOverCut = sigma / cutoff
    sig6OverCut6 = sigOverCut ** 6.0
    sig12OverCut12 = sig6OverCut6 * sig6OverCut6

    engShift = -4.0 * epsilon * (sig12OverCut12 - sig6OverCut6)

    print("test_EvaluatorLJ_energy is not yet implemented")


def test_EvaluatorLJ_energy_device(make_state):
    """ analytically verifies that the EvaluatorLJ returns proper energy on the GPU """
    state = make_state['state']
    # sigma, epsilon, cutoff
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0

    print("test_EvaluatorLJ_energy_device is not yet implemented")


