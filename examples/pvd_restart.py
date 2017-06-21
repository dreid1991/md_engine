import sys
sys.path.append('..//build/python/build/lib.linux-x86_64-2.7')
from DASH import *
from math import *
import re
import argparse
import matplotlib.pyplot as plt
from random import random

state = State()
#state.deviceManager.setDevice(0)
state.periodicInterval = 7

state.readConfig.loadFile('pvd.xml')
state.readConfig.next()

f2d = Fix2d(state, handle='2d', applyEvery=10)
state.activateFix(f2d)

ljcut = FixLJCut(state, handle='ljcut')

state.activateFix(ljcut)

fixSpring = FixSpringStatic(state, handle='substrateSpring', groupHandle='sub')
state.activateFix(fixSpring)


#okay, substrate is set up, going to do deposition

wallDist = 10
topWall = FixWallHarmonic(state, handle='wall', groupHandle='all',
                          origin=Vector(0, state.bounds.hi[1], 0),
                          forceDir=Vector(0, -1, 0), dist=wallDist, k=10)

state.activateFix(topWall)
subTemp = 0.14
fixNVT = FixNoseHoover(state, handle='nvt', groupHandle='sub', temp=subTemp, timeConstant=0.1)
state.activateFix(fixNVT)


print state.dt
#for a in state.atoms:
    #print a.type
integrator = IntegratorVerlet(state)
integrator.run(10000)
print 'FINISHED SECOND RUN'


# Start deposition
toDeposit = [(4*26, 4*14)]
depositionRuns = 600
newVaporGroup = 'vapor'
vaporTemp = 1.0
toDepIdx = 0
state.createGroup('film')
for i in range(depositionRuns):

    writer = WriteConfig(state, handle='writer', fn='pvd_%d' % i, format='xyz', writeEvery=10000)
    state.activateWriteConfig(writer)
    print 'Deposition step {}'.format(i)
    maxY = max(a.pos[1] for a in state.atoms)
    print 'max y'
    print maxY
    newTop = max(maxY + 15 + wallDist, state.bounds.hi[1])
    hi = state.bounds.hi
    hi[1] = newTop
    state.bounds.hi = hi
    topWall.origin = Vector(0, state.bounds.hi[1], 0)
    #state.bounds.hi[1] = newTop #changing the bounds as the film grows
    #state.bounds.trace[1] = state.bounds.hi[1] - state.bounds.lo[1]
    #topWall.origin[1] = newTop #moving the harmonic bounding wall
    print('Wall y-pos: %f' % newTop)
    populateBounds = Bounds(state,
                            lo=Vector(state.bounds.lo[0], newTop-wallDist-5, 0),
                            hi=Vector(state.bounds.hi[0], newTop-wallDist, 0))
    InitializeAtoms.populateRand(state, bounds=populateBounds, handle='type1',
                                 n=toDeposit[toDepIdx][0], distMin=1)
    InitializeAtoms.populateRand(state, bounds=populateBounds, handle='type2',
                                 n=toDeposit[toDepIdx][1], distMin=1)

    newAtoms = []
    for k in range(1, 1+sum(toDeposit[toDepIdx])):
        na = state.atoms[-k]
        newAtoms.append(state.atoms[-k])
        print('New atom: {}, pos ({},{},{})'.format(na.id, na.pos[0], na.pos[1], na.pos[2]))

    state.createGroup(newVaporGroup)
    state.addToGroup(newVaporGroup, [a.id for a in newAtoms])
    state.addToGroup('film', [a.id for a in newAtoms])
    InitializeAtoms.initTemp(state, newVaporGroup, vaporTemp)
    state.destroyGroup(newVaporGroup)
    for a in newAtoms:
        a.vel[1] = min(-abs(a.vel[1]), -0.2)
    curTemp = sum([a.vel.lenSqr()/3.0 for a in newAtoms]) / len(newAtoms)
    for a in newAtoms:
        a.vel *= sqrt(vaporTemp / curTemp)

    integrator.run(1000000)
    toDepIdx += 1
    toDepIdx = toDepIdx % len(toDeposit)

    curTemp = sum([a.vel.lenSqr()/2.0 for a in state.atoms]) / len(state.atoms)
    print 'cur temp %f ' % curTemp
    state.deactivateWriteConfig(writer)
    print 'avg eng %f' % integrator.energyAverage('film')


