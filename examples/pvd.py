import sys
sys.path = sys.path + [ '/home/julian/cornballMD/build/python/build/lib.linux-x86_64-2.7' ]
from Sim import *
from math import *
from random import random

state = State()
f2d = Fix2d(state, handle='2d', applyEvery=10)
#state.deviceManager.setDevice(0)
state.periodicInterval = 1
state.shoutEvery = 50000 #how often is says % done
state.is2d = True
state.setPeriodic(2, False)
state.rCut = 2.5 #need to implement padding
state.padding = 0.5
state.seedRNG()

# z bounds taken care of automatically in 2d simulation
state.bounds = Bounds(state, lo=Vector(0, 0, -4),
                             hi=Vector(30, 70.0, 4))
state.grid = AtomGrid(state, dx=3.5, dy=3.5, dz=3) #as is dz
state.atomParams.addSpecies(handle='substrate', mass=1)
state.atomParams.addSpecies(handle='type1', mass=1)
state.atomParams.addSpecies(handle='type2', mass=1)



#state.activateFix(f2d)

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
                             handle='substrate', n=212, distMin = 0.6)

subTemp = 0.14
InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments
state.createGroup('sub')
state.addToGroup('sub', list(state.atoms))



def springFunc(id, pos):
    pos[1] = (substrateInitBounds.lo[1] + substrateInitBounds.hi[1]) / 2
    return pos

def springFuncEquiled(id, pos):
    return pos

fixSpring = FixSpringStatic(state, handle='substrateSpring', groupHandle='sub',
                            k=1, tetherFunc=springFunc,
                            multiplier=Vector(0.5, 1, 0))
state.activateFix(fixSpring)

print("Creating Nose Hoover thermostat")
fixNVT = FixNoseHoover(state, handle='nvt', groupHandle='sub', temp=subTemp, timeConstant=0.1)
#fixNVT = FixNVTRescale(state, handle='nvt', groupHandle='sub',
#                       intervals=[0, 1], temps=[subTemp, subTemp],
#                       applyEvery = 10)
print("Activating Nose Hoover thermostat")
state.activateFix(fixNVT)

# Start simulation
writer = WriteConfig(state, handle='writer', fn='pvd_test', format='xyz',
                     writeEvery=2000)
state.activateWriteConfig(writer)

state.dt = 0.001
integrator = IntegratorVerlet(state)
integrator.run(100) #letting substrate relax
exit()
integratorRelax = IntegratorRelax(state)
#integratorRelax.set_params(dtMax_mult=1);
integratorRelax.run(50000, 1e-5)
print 'FINISHED FIRST RUN'
state.dt = 0.005

fixSpring.tetherFunc = springFuncEquiled
fixSpring.updateTethers() #tethering to the positions they fell into
fixSpring.k = 1000
fixSpring.multiplier = Vector(1, 1, 0) #now spring holds in both dimensions

InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments
#okay, substrate is set up, going to do deposition
integrator.run(10000)
print 'FINISHED SECOND RUN'

print("Average per particle energy: {}".format(integratorRelax.energyAverage()))

# Start deposition
toDeposit = [(7, 3), (6, 4)]
depositionRuns = 20

wallDist = 10
topWall = FixWallHarmonic(state, handle='wall', groupHandle='all',
                          origin=Vector(0, state.bounds.hi[1], 0),
                          forceDir=Vector(0, -1, 0), dist=wallDist, k=10)

state.activateFix(topWall)

newVaporGroup = 'vapor'
vaporTemp = 1.0
toDepIdx = 0

for i in range(depositionRuns):
    print 'Deposition step {}'.format(i)
    maxY = max(a.pos[1] for a in state.atoms)
    newTop = max(maxY + 15 + wallDist, state.bounds.hi[1])
    state.bounds.hi[1] = newTop #changing the bounds as the film grows
    state.bounds.trace[1] = state.bounds.hi[1] - state.bounds.lo[1]
    topWall.origin[1] = newTop #moving the harmonic bounding wall
    print("Wall y-pos: {}".format(newTop))
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
        print("New atom: {}, pos ({},{},{})".format(na.id, na.pos[0], na.pos[1], na.pos[2]))

    state.createGroup(newVaporGroup)
    state.addToGroup(newVaporGroup, newAtoms)
    InitializeAtoms.initTemp(state, newVaporGroup, vaporTemp)
    state.destroyGroup(newVaporGroup)
    for a in newAtoms:
        a.vel[1] = min(-abs(a.vel[1]), -0.2)
    curTemp = sum([a.vel.lenSqr()/3.0 for a in newAtoms]) / len(newAtoms)
    print vaporTemp
    print curTemp
    for a in newAtoms:
        a.vel *= sqrt(vaporTemp / curTemp)

    integrator.run(86000)
    toDepIdx += 1
    toDepIdx = toDepIdx % len(toDeposit)

    curTemp = sum([a.vel.lenSqr()/3.0 for a in state.atoms]) / len(state.atoms)
    print 'cur temp %f ' % curTemp

#and we're done!

state.dt = 0.001
integratorRelax.run(20000, 1e-5)
print("Average per particle energy: {}".format(integratorRelax.energyAverage()))

