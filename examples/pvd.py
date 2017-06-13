import sys
sys.path.append('../build/python/build/lib.linux-x86_64-2.7')
from DASH import *
from math import *
import re
import argparse
import matplotlib.pyplot as plt
from random import random

state = State()
#state.deviceManager.setDevice(0)
state.periodicInterval = 7
state.shoutEvery = 50000 #how often is says % done
state.is2d = True
state.setPeriodic(2, False)
state.rCut = 2.5 #need to implement padding
state.padding = 0.5
state.seedRNG()

# z bounds taken care of automatically in 2d simulation
state.bounds = Bounds(state, lo=Vector(0, 0, -4),
                             hi=Vector(2*240, 20.0, 4))
state.atomParams.addSpecies(handle='substrate', mass=1)
state.atomParams.addSpecies(handle='type1', mass=1)
state.atomParams.addSpecies(handle='type2', mass=1)


f2d = Fix2d(state, handle='2d', applyEvery=10)
state.activateFix(f2d)

ljcut = FixLJCut(state, handle='ljcut')

state.activateFix(ljcut)

#deposited atom iteraction parameters
ljcut.setParameter(param='eps', handleA='type1', handleB='type1', val=1)
ljcut.setParameter(param='sig', handleA='type1', handleB='type1', val=1)

ljcut.setParameter(param='eps', handleA='type1', handleB='type2', val=1.5)
ljcut.setParameter(param='sig', handleA='type1', handleB='type2', val=0.8)

ljcut.setParameter(param='eps', handleA='type2', handleB='type2', val=0.5)
ljcut.setParameter(param='sig', handleA='type2', handleB='type2', val=0.88)

#substrate interaction parameters
ljcut.setParameter(param='eps', handleA='type1', handleB='substrate', val=1.0)
ljcut.setParameter(param='sig', handleA='type1', handleB='substrate', val=0.75)
ljcut.setParameter(param='eps', handleA='type2', handleB='substrate', val=1.0)
ljcut.setParameter(param='sig', handleA='type2', handleB='substrate', val=0.7)
ljcut.setParameter(param='eps', handleA='substrate', handleB='substrate', val=0.1)
ljcut.setParameter(param='sig', handleA='substrate', handleB='substrate', val=0.6)

#deposit substrate
substrateInitBounds = Bounds(state, lo=Vector(state.bounds.lo[0], 2.5, 0),
                                    hi=Vector(state.bounds.hi[0], 9.5, 0))
InitializeAtoms.populateRand(state, bounds=substrateInitBounds,
                             handle='substrate', n=4*848, distMin = 0.6)

subTemp = 0.14
InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments
state.createGroup('sub')
state.addToGroup('sub', [a.id for a in state.atoms])



def springFunc(id, pos):
    pos[1] = (substrateInitBounds.lo[1] + substrateInitBounds.hi[1]) / 2
    return pos

def springFuncEquiled(id, pos):
    return pos

fixSpring = FixSpringStatic(state, handle='substrateSpring', groupHandle='sub',
                            k=0.1, tetherFunc=springFunc,
                            multiplier=Vector(0.5, 1, 0))
state.activateFix(fixSpring)

print('Creating Nose Hoover thermostat')
fixNVT = FixNoseHoover(state, handle='nvt', groupHandle='sub', temp=subTemp, timeConstant=0.1)
print('Activating Nose Hoover thermostat')
state.activateFix(fixNVT)

# Start simulation

state.dt = 0.001
integrator = IntegratorVerlet(state)
#integrator.run(100) #letting substrate relax

integratorRelax = IntegratorGradientDescept(state)
#integratorRelax.set_params(dtMax_mult=1);
#integratorRelax.run(250000, 1e-5)
integratorRelax.run(250000, 1)
fixSpring.k = 1.0
integratorRelax.run(25000, 1)
print 'FINISHED FIRST RUN'
state.dt = 0.005

InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments

fixSpring.tetherFunc = springFuncEquiled
fixSpring.updateTethers() #tethering to the positions they fell into
fixSpring.k = 1000
fixSpring.multiplier = Vector(1, 1, 0) #now spring holds in both dimensions

InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments
writerXML = WriteConfig(state, handle='writerXML', fn='pvd', format='xml', writeEvery=10)
writerXML.write()
#state.activateWriteConfig(writerXML)
#okay, substrate is set up, going to do deposition

wallDist = 10
topWall = FixWallHarmonic(state, handle='wall', groupHandle='all',
                          origin=Vector(0, state.bounds.hi[1], 0),
                          forceDir=Vector(0, -1, 0), dist=wallDist, k=10)

state.activateFix(topWall)


integrator.run(10000)
print 'FINISHED SECOND RUN'

print('Average per particle energy: %f' % integratorRelax.energyAverage())

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
    writerXML.write()
    print 'avg eng %f' % integrator.energyAverage('film')

#and we're done!

state.dt = 0.001

