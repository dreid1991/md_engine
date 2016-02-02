import sys
sys.path.append('/home/daniel/Documents/auxetic/lib')
from Sim import *
from random import random

state = State()
state.periodicInterval = 10
state.is2d = True #could wrap such that setting 2d automatically makes system not periodic in Z
state.setPeriodic(2, False)
state.rCut = 3.0 #need to implement padding

state.bounds = Bounds(state, lo=Vector(0, 0, 0), hi=Vector(120, 30, 0))# z bounds taken care of automatically in 2d simulation
state.grid = AtomGrid(state, dx=4, dy=4, dz=1) #as is dz
state.atomParams.addSpecies(handle='substrate', mass=1)
state.atomParams.addSpecies(handle='type1', mass=1)
state.atomParams.addSpecies(handle='type2', mass=1)


f2d = Fix2d(state, handle='2d', applyEvery=10)
state.activateFix(f2d)

ljcut = FixLJCut(state, handle='ljcut', groupHandle='all')

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
#ljcut.setParameter(param='eps', handleA='substrate', handleB='substrate', val=0.1)
#ljcut.setParameter(param='sig', handleA='substrate', handleB='substrate', val=0.6)
ljcut.setParameter(param='eps', handleA='substrate', handleB='substrate', val=.1)
ljcut.setParameter(param='sig', handleA='substrate', handleB='substrate', val=.6)
substrateInitBounds = Bounds(state, lo=Vector(0, 4.5, 0), hi=Vector(120, 7.5, 0))
InitializeAtoms.populateRand(state, bounds=substrateInitBounds, handle='substrate', n=250, distMin = 0.6)

InitializeAtoms.initTemp(state, 'all', 0.16) #need to add keyword arguments
state.createGroup('sub')
state.addToGroup('sub', list(state.atoms)) #adding all the atoms (the substrate) to substrate group



writer = WriteConfig(state, handle='writer', fn='pvd_test', format='base64', writeEvery=1000) #can write in plain text or base64 for higher accuracy.  Both in xml.  Files can be read by simulation.  I wrote a script to render these using matplotlib, should movie.py in the files I sent
state.activateWriteConfig(writer)

def springFunc(id, pos):
    pos[1] = (substrateInitBounds.lo[1] + substrateInitBounds.hi[1]) / 2
    return pos

def springFuncEquiled(id, pos):
    return pos

fixSpring = FixSpringStatic(state, handle='substrateSpring', groupHandle='sub', k=10, tetherFunc=springFunc, multiplier=Vector(0, 1, 0)) #multiplier multiplies self by force for each tether.  So this is a spring that applies only in y
state.activateFix(fixSpring)

fixNVT = FixNVTRescale(state, handle='nvt', groupHandle='all', intervals=[0, 1], temps=[.12, .12], applyEvery = 10) #continues to iterate at final temperature after gettings to its last interval, so still 0.12
state.activateFix(fixNVT)
integrater = IntegraterVerlet(state)
integrater.run(10000) #letting substrate relax
print 'FINISHED FIRST RUN'
fixSpring.tetherFunc = springFuncEquiled
fixSpring.updateTethers() #tethering to the positions they fell into
fixSpring.k = 500
fixSpring.multiplier = Vector(1, 1, 0) #now spring holds in both dimensions
#okay, substrate is set up, going to do deposition
integrater.run(10000)
toDeposit = (26, 14) #how many of types 1, 2, I want to deposit each time
depositionRuns = 80

wallDist = 10
topWall = FixWallHarmonic(state, handle='wall', groupHandle='all', origin=Vector(0, state.bounds.hi[1], 0), forceDir=Vector(0, -1, 0), dist=wallDist, k=10)

state.activateFix(topWall)

state.shoutEvery=50000 #how often is says % done
newVaporGroup = 'vapor'
vaporTemp = 1.0
for i in range(depositionRuns):
    maxY = max(a.pos[1] for a in state.atoms)
    newTop = min(max(maxY + 10 + wallDist, state.bounds.hi[1]), state.bounds.hi[1]+1)
    state.bounds.hi[1] = newTop #changing the bounds as the film grows
    topWall.origin[1] = newTop #moving the harmonic bounding wall
    populateBounds = Bounds(state, lo=Vector(state.bounds.lo[0], newTop-wallDist, 0), hi=Vector(state.bounds.hi[0], newTop-wallDist+5, 0))
    InitializeAtoms.populateRand(state, bounds=populateBounds, handle='type1', n=toDeposit[0], distMin=1)
    InitializeAtoms.populateRand(state, bounds=populateBounds, handle='type2', n=toDeposit[1], distMin=1)

    newAtoms = []
    for i in range(sum(toDeposit)):
        newAtoms.append(state.atoms[-i]) #just slicing a boost-wrapped vector copies the atom objects rather than taking pointers, can't do that (could if I worked by id rather than pointer though)

    state.createGroup(newVaporGroup)
    state.addToGroup(newVaporGroup, newAtoms)
    InitializeAtoms.initTemp(state, 'vapor', vaporTemp)
    state.destroyGroup(newVaporGroup)
    for a in newAtoms:
        a.vel[1] = min(-abs(a.vel[1]), -0.2)
    #print '%d %d %d' % (len([a for a in state.atoms if a.type==0]), len([a for a in state.atoms if a.type==1]), len([a for a in state.atoms if a.type==2]))
    integrater.run(70000) #very short, just for testing


#and we're done!



