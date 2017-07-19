from DASH import *
from math import *
import numpy as np

def getRandomOrientation():
    # randomly rotates the molecule in 3D space
    # -- get a vector of norm 1 pointing in some random direction
    x1 = 1.0 - 2.0*np.random.random()
    x2 = 1.0 - 2.0*np.random.random()
    x3 = 1.0 - 2.0*np.random.random()
    orientation = Vector(x1,x2,x3).normalized()
    return orientation

def rotateBy(origin, bond, theta):
    # all of the rigid water models are isosceles triangles
    axis = bond.normalized()
    axis = np.asarray((axis[0], axis[1], axis[2]))

    # save this as our basis1
    basis1 = axis
    # -- the dot product of two orthogonal vectors is zero
    #    so, a*x + b*y + c*z = 0 ---> c = -(a*x - b*y) / z
    basis2 = np.asarray((1.0-2.0*np.random.random(),
                         1.0-2.0*np.random.random(),0))
    c  = -np.dot(basis2,axis) / axis[2]

    # here is our orthogonal vector. needs to be normalized though.
    solution = np.asarray((basis2[0],basis2[1],c))

    # normalized. this is an orthogonal vector to our basis vector 'axis'
    basis2 = solution / np.linalg.norm(solution)

    # choose one of two bases about which we can rotate by theta
    randomAxis = np.random.random();
    if (randomAxis < 0.5) :
        axis = basis2
    else:
        axis = np.cross(basis1,basis2)
    # set up our rotation matrix, using 'axis' as our new basis. assume origin (the
    # position of our oxygen atom) is simply (0,0,0); its actual location doesnt matter
    # --- we'll use 'origin' later to get the actual position of the Hydrogen
    rotationMatrix = np.zeros((3,3))

    rotationMatrix[0][0] = cos(theta) + axis[0]*axis[0]*(1-cos(theta))
    rotationMatrix[0][1] = axis[0]*axis[1]*(1-cos(theta)) - axis[2]*sin(theta)
    rotationMatrix[0][2] = axis[0]*axis[2]*(1-cos(theta)) + axis[1]*sin(theta)

    rotationMatrix[1][0] = axis[1]*axis[0]*(1-cos(theta)) + axis[2]*sin(theta)
    rotationMatrix[1][1] = cos(theta) + axis[1]*axis[1]*(1-cos(theta))
    rotationMatrix[1][2] = axis[1]*axis[2]*(1-cos(theta)) - axis[0]*sin(theta)

    rotationMatrix[2][0] = axis[2]*axis[0]*(1-cos(theta)) - axis[1]*sin(theta)
    rotationMatrix[2][1] = axis[2]*axis[1]*(1-cos(theta)) + axis[0]*sin(theta)
    rotationMatrix[2][2] = cos(theta) + axis[2]*axis[2]*(1-cos(theta))

    rotation = np.dot(rotationMatrix,basis1)
    rotation *= bond.len()

    # and convert from np.ndarray to Vector type
    position = origin + Vector(rotation[0], rotation[1], rotation[2])

    return position

# 'bondLength' defaults to Real units
# pass bondLength/sigma(O) to scale for LJ simulation
def create_TIP3P(state, oxygenHandle, hydrogenHandle, bondLength = 0.9572, center=None, orientation=None):
    if center==None:
        center = state.Vector(0, 0, 0)
    if orientation==None:
        state.addAtom(handle=  oxygenHandle, pos=center,q=-0.8340)
        h1Pos = center + Vector(bondLength, 0, 0)
        state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.4170)
        theta = 1.824218134
        h2Pos = center + Vector(cos(theta), sin(theta), 0) * bondLength
        state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.4170)
        return state.createMolecule([state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])
    elif orientation=="random":
        state.addAtom(handle=oxygenHandle, pos=center, q=-0.8340)
        offsetH1 = getRandomOrientation() * bondLength
        h1Pos = center + offsetH1
        state.addAtom(handle=hydrogenHandle,pos=h1Pos,q=0.4170)
        theta = 1.824218134
        # the direction from oxygen to h1Pos is just Offset
        # so, we provide the position of the oxygen, the length of the O-H bond, and the angle HOH;
        # this defines a given water model
        h2Pos = rotateBy(center,offsetH1,theta)
        # print 'h2-h1 bond length: ', (h2Pos - h1Pos).len()
        # print 'h2-O  bond length: ', (h2Pos - center).len()
        # print 'h1-O  bond length: ', (h1Pos - center).len()
        state.addAtom(handle=hydrogenHandle,pos=h2Pos,q=0.4170)
        return state.createMolecule([state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])


def create_TIP3P_long(state, oxygenHandle, hydrogenHandle, center=None):
    if center==None:
        center = state.Vector(0, 0, 0)
    state.addAtom(handle=oxygenHandle, pos=center, q=-0.83)
    h1Pos = center + Vector(0.9572, 0, 0)
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.415)
    theta = 1.824218134
    h2Pos = center + Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.4170)
    return state.createMolecule([state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

def create_TIP4P(state, oxygenHandle, hydrogenHandle, mSiteHandle, center=None):
    if center==None:
        center = state.Vector(0, 0, 0)
    state.addAtom(handle=oxygenHandle, pos=center, q=0)
    offset1 = Vector(0.9572, 0, 0)
    h1Pos = center + offset1
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.52)
    theta = 1.824218134
    offset2 = Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    h2Pos = center + offset2
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.52)
    mSiteOffset = (offset1 + offset2).normalized() * 0.1546

    state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=-1.04)
    return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

def create_TIP4P_long(state, oxygenHandle, hydrogenHandle, mSiteHandle, center=None):
    if center==None:
        center = state.Vector(0, 0, 0)
    state.addAtom(handle=oxygenHandle, pos=center, q=0)
    offset1 = Vector(0.9572, 0, 0)
    h1Pos = center + offset1
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.5242)
    theta = 1.824218134
    offset2 = Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    h2Pos = center + offset2
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.5242)
    mSiteOffset = (offset1 + offset2).normalized() * 0.1546

    state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=-1.0484)
    return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

def create_TIP4P_2005(state, oxygenHandle, hydrogenHandle, mSiteHandle, center=None, orientation=None):
    if center==None:
        center = state.Vector(0, 0, 0)
    if orientation==None:
        state.addAtom(handle=oxygenHandle, pos=center, q=0)
        offset1 = Vector(0.9572, 0, 0)
        h1Pos = center + offset1
        state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.5897)
        theta = 1.824218134
        offset2 = Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
        h2Pos = center + offset2
        state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.5897)
        mSiteOffset = (offset1 + offset2).normalized() * 0.1546

        state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=-1.1794)
        return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

    elif orientation=="random":
        state.addAtom(handle=oxygenHandle, pos=center, q=0)
        offset1 = getRandomOrientation() * 0.9572
        h1Pos = center + offset1
        state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.5897)
        theta = 1.824218134
        h2Pos = rotateBy(center,offset1,theta)
        offset2 = h2Pos - center
        state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.5897)
        mSiteOffset = (offset1 + offset2).normalized() * 0.1546
        state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=-1.1794)
        return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

# q-TIP4P/F - see J. Chem. Phys. 131 024501 (2009)
def create_TIP4P_Flexible(state, oxygenHandle, hydrogenHandle, mSiteHandle, center=None, orientation=None):
    if (center==None):
        center = state.Vector(0,0,0)
    # the charges on each site
    qO = 0.0
    qM = -1.1128
    qH = -0.5 * qM
    theta = 1.8744836 # radians
    rH = 0.9419 # equilibrium OH bond length for this model
    rOM = 0.147144032 # calculated separately
    if (orientation==None):
        state.addAtom(handle=oxygenHandle, pos=center, q=qO)
        offset1 = Vector(1.0, 0, 0) * rH
        h1Pos = center + offset1
        state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=qH)
        offset2 = Vector(cos(theta), sin(theta), 0) * rH
        h2Pos = center + offset2
        state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=qH)
        mSiteOffset = (offset1 + offset2).normalized() * rOM
        state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=qM)
        return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

    elif orientation=="random":
        # add the oxygen at position 'center'
        state.addAtom(handle=oxygenHandle, pos=center, q=qO)
        # get a random vector - this will be the axis along which we add the first Hydrogen at the corresponding distance rOH
        offset1 = getRandomOrientation() * rH
        # the position is simply center + offset1
        h1Pos = center + offset1
        # add the atom
        state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=qH)
        # now, rotate around the axis of the first hydrogen by theta and place the second hydrogen there
        h2Pos = rotateBy(center,offset1,theta)
        offset2 = h2Pos - center
        state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=qH)
        # finally, bisect the angle formed by the hydrogens, and place the M-site at a distance rOM from the oxygen atom
        mSiteOffset = (offset1 + offset2).normalized() * rOM
        state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=qM)
        # and our returned molecule
        return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])
