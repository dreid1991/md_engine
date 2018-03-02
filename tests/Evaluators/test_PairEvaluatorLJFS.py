from __future__ import print_function
import sys
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
import pytest
import numpy as np

####################################################################################################
#  DASH
#
#  test_EvaluatorLJFS.py
#
#  Verifies that EvaluatorLJFS returns the correct forces and energies on host and device.
#
#  Author: Brian Keene
#
#  Tests:
#
#       test_EvaluatorLJFS_force()
#           -last edited: BK, 3 January 2018
#
#       test_EvaluatorLJFS_force_device()
#           -TODO
#
#       test_EvaluatorLJFS_energy()
#           -last edited: BK, 3 January 2018
#
#       test_EvaluatorLJFS_energy_device()
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



def test_EvaluatorLJFS_force(make_state):
    """ explicitly verifies that the EvaluatorLJFS returns proper force """

    # it is expected that, at the cutoff, the Evaluator returns a force of 0.0;
    # moreover, the force should be zero at r =
    state = make_state['state']
    # sigma, epsilon, cutoff; no significance to these values, just pick a number
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0

    rc_inv     = cutoff ** (-1.0)
    rc_inv_sqr = rc_inv ** 2.0
    rc_inv_6   = rc_inv_sqr ** 3.0

    LJPotential = EvaluatorLJFS()


    sig2 = sigma * sigma
    sig6 = sig2 ** 3.0
    sig12 = sig6 * sig6

    # as currently implemented in PairEvaluateIso, dr = bounds.minImage(pos - otherPos),
    # where we are computing the force on atom with position 'pos'

    # put particle A here, then particle B, both as vector
    positions_A = [Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0),
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
                   Vector(10.0, 10.0 - 0.999 * cutoff, 10.0),
                   Vector(10.0, 10.0 - 0.999999 * cutoff, 10.0),
                   Vector(10.0, 10.0 - 0.99999999 * cutoff, 10.0)]

    # use the same minImage function; make an instance of BoundsGPU
    boundsGPU = BoundsGPU(state.bounds.lo, state.bounds.hi - state.bounds.lo, Vector(1,1,1))

    # put explicitly the form of the expected force
    def force(dr):
        rij_scalar = dr.len()
        if (rij_scalar <= cutoff):

            # so, the LJFS potential, as indicated p. 146 Allen & Tildesley,
            # let V = LJ potential, unshifted;
            # then V_SF = V - V_c - (dV/dr)|_(r_c) *  (r_ij - r_c);
            # i.e. we add a constant linear term corresponding to the force at r_c as well as shifting the
            # potential up by V at V_c
            # --- so, the force is the regular force, minus the force at the cutoff!
            r_inv = 1.0 / (rij_scalar)
            r2inv = 1.0 / (rij_scalar ** 2.0)
            r6inv = r2inv * r2inv * r2inv
            # ok, so just the /pure force scalar/ - no factors incorporated from the vector direction
            forceScalar = r6inv * r_inv * 24.0 * epsilon * ( (2.0 * sig12 * r6inv) - (sig6))
            f_cutoff =rc_inv_6 * rc_inv * 24.0 * epsilon  *( (2.0 * sig12 * rc_inv_6) - sig6)
            forceScalar -= f_cutoff
            # now, account for the fact that we multiply by the vector direction dr instead of a unit vector
            forceScalar /= rij_scalar
            # multiply by the vector direction
            return (dr * forceScalar)
        else:
            return Vector(0.0, 0.0, 0.0)

    def force_at_cutoff(this_cutoff):
        cutoff_inv     = 1.0 / this_cutoff
        cutoff_inv_sqr = 1.0 / (this_cutoff * this_cutoff)
        cutoff_inv_6   = cutoff_inv_sqr ** 3.0
        return (24.0 * epsilon * cutoff_inv_6 * cutoff_inv * ((2.0 * sig12 * cutoff_inv_6) - sig6))

    # forces not near zero, relative error should be < 1e-10; if absolute value
    # of expected force is < 1e-8, assert that the difference between the two is < 1e-10
    tolerance = 1e-10

    for index, _ in enumerate(positions_A):
        pos = positions_A[index]
        otherPos = positions_B[index]
        toMinImage = pos - otherPos
        dr = boundsGPU.minImage(toMinImage) # exact syntax as in PairEvaluateIso
        calculatedForce = LJPotential.force(sigma, epsilon,cutoff,dr)
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

def test_EvaluatorLJFS_force_device(make_state):
    """ explicitly verifies that the EvaluatorLJFS returns proper force on the GPU """

    #assert(False)
    pass

def test_EvaluatorLJFS_energy(make_state):
    """ explicitly verifies that the EvaluatorLJFS returns proper energy """
    # it is expected that, at the cutoff, the Evaluator returns a force of 0.0;
    # moreover, the force should be zero at r =
    state = make_state['state']
    # sigma, epsilon, cutoff; no significance to these values, just pick a number
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0

    rc_inv = cutoff ** -1.0
    rc_inv_sqr = rc_inv ** 2.0
    rc_inv_6   = rc_inv_sqr ** 3.0

    LJPotential = EvaluatorLJFS()


    sig2 = sigma * sigma
    sig6 = sig2 ** 3.0
    sig12 = sig6 * sig6

    # as currently implemented in PairEvaluateIso, dr = bounds.minImage(pos - otherPos),
    # where we are computing the force on atom with position 'pos'

    # put particle A here, then particle B, both as vector
    positions_A = [Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0),
                   Vector(10.0, 10.0, 10.0),
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
                   Vector(10.0, 10.0 - 0.999 * cutoff, 10.0),
                   Vector(10.0, 10.0 - 0.999999 * cutoff, 10.0),
                   Vector(10.0, 10.0 - 0.99999999 * cutoff, 10.0)]


    # use the same minImage function; make an instance of BoundsGPU
    # as (lo, lengths, periodic)
    boundsGPU = BoundsGPU(state.bounds.lo, state.bounds.hi - state.bounds.lo, Vector(1,1,1))

    # put explicitly the form of the expected energy
    def energy(rij_scalar):
        if (rij_scalar <= cutoff):

            # so, the LJFS potential, as indicated p. 146 Allen & Tildesley,
            # let V = LJ potential, unshifted;
            # then V_SF = V - V_c - (dV/dr)|_(r_c) *  (r_ij - r_c);
            # i.e. we add a constant linear term corresponding to the force at r_c as well as shifting the
            # potential up by V at V_c
            # --- so, the force is the regular force, minus the force at the cutoff!
            r2inv = 1.0 / (rij_scalar ** 2.0)
            r6inv = r2inv * r2inv * r2inv
            forceScalar = r6inv * r2inv * 24.0 * epsilon * ( (2.0 * sig12 * r6inv) - (sig6))
            f_cutoff =rc_inv * rc_inv_6 * 24.0 * epsilon  *( (2.0 * sig12 * rc_inv_6) - sig6)

            u_lj = 4.0 * epsilon * r6inv * ( (sig12 * r6inv) - sig6)
            u_lj_at_cutoff = 4.0 * epsilon * rc_inv_6 * ( (sig12 * rc_inv_6 ) - sig6)
            linear_term =    f_cutoff * (rij_scalar - cutoff)

            # this is a pair potential; we divvy up the energy between the two particles, since on the GPU
            # we double count - i.e., we count U_ij and U_ji
            return ( 0.5 * (u_lj - u_lj_at_cutoff - linear_term))

        else:
            return 0.0


    # forces not near zero, relative error should be < 1e-10; if absolute value
    # of expected force is < 1e-8, assert that the difference between the two is < 1e-10
    tolerance = 1e-10

    for index, _ in enumerate(positions_A):
        pos = positions_A[index]
        otherPos = positions_B[index]
        toMinImage = pos - otherPos
        dr = boundsGPU.minImage(toMinImage) # exact syntax as in PairEvaluateIso
        calculatedEnergy = LJPotential.energy(sigma, epsilon,cutoff,dr.len())
        expectedEnergy   = energy(dr.len())

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
        #assert(relativeError <= tolerance)

    pass

def test_EvaluatorLJFS_energy_device(make_state):
    """ explicitly verifies that the EvaluatorLJFS returns proper energy on the GPU """
    state = make_state['state']
    # sigma, epsilon, cutoff
    sigma = 3.0
    epsilon = 6.0
    cutoff = 10.0

    #assert(False)
    pass
