from DASH import *
from math import *
import numpy as np

def getRandomOrientation():
    # randomly rotates the molecule in 3D space
    # -- get a vector of norm 1 pointing in some random direction
    x1 = np.random.random()
    x2 = np.random.random()
    x3 = np.random.random()
    orientation = Vector(x1,x2,x3).normalized()
    return orientation

def rotateBy(origin, bond, theta):
    # all of the rigid water models are isosceles triangles
    axis = bond.normalized()
    axis = np.asarray((axis[0], axis[1], axis[2]))

    # save this as our basis1
    basis1 = axis
    # we have two degrees of freedom.. we'll solve for c
    # -- the dot product of two orthogonal vectors is zero
    #    so, a*x + b*y + c*z = 0 ---> c = -(a*x - b*y) / z
    basis2 = np.asarray((1,1,0))
    c  = -np.dot(basis2,axis) / axis[2]

    # here is our orthogonal vector. needs to be normalized though.
    solution = np.asarray((1,1,c))

    # normalized. this is an orthogonal vector to our basis vector 'axis'
    basis2 = solution / np.linalg.norm(solution)

    # this is that axis about which we will rotate
    axis = basis2
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
    position = origin + Vector(rotation[0], rotation[1], rotation[2])

    return position

def create_TIP3P(state, oxygenHandle, hydrogenHandle, center=None, orientation=None):
    if center==None:
        center = state.Vector(0, 0, 0)
    if orientation==None:
        state.addAtom(handle=  oxygenHandle, pos=center,q=-0.8340)
        h1Pos = center + Vector(0.9572, 0, 0)
        state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.4170)
        theta = 1.824218134
        h2Pos = center + Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
        state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.4170)
        return state.createMolecule([state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])
    elif orientation=="random":
        state.addAtom(handle=oxygenHandle, pos=center, q=-0.8340)
        offsetH1 = getRandomOrientation() * 0.9572
        h1Pos = center + offsetH1
        state.addAtom(handle=hydrogenHandle,pos=h1Pos,q=0.4170)
        theta = 1.824218134
        # the direction from oxygen to h1Pos is just Offset
        # so, we provide the position of the oxygen, the length of the O-H bond, and the angle HOH;
        # this defines a given water model
        h2Pos = rotateBy(center,offsetH1,theta)
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
        orientation = Vector(1.0, 1.0, 1.0)
    else:
        orientation = Vector(1.0, 1.0, 1.0)
        orientation = getRandomOrientation();
    center *= orientation
    state.addAtom(handle=oxygenHandle, pos=center, q=0)
    offset1 = Vector(0.9572, 0, 0)
    h1Pos = center + offset1
    h1Pos *= orientation
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.5897)
    theta = 1.824218134
    offset2 = Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    h2Pos = center + offset2
    h2Pos *= orientation
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.5897)
    mSiteOffset = (offset1 + offset2).normalized() * 0.1546

    state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=-1.1794)
    return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])


