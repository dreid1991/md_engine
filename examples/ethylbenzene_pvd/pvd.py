import sys
sys.path = sys.path + [ '../../build/python/build/lib.linux-x86_64-2.7', '../../util_py' ]
from LAMMPS_Reader import LAMMPS_Reader
from math import *
from random import random
from Sim import *

state = State()
#state.deviceManager.setDevice(0)
state.periodicInterval = 7
state.shoutEvery = 100 #how often is says % done
state.rCut = 3.0
state.padding = 0.6
state.seedRNG()

#state.deviceManager.setDevice(0)
state.bounds = Bounds(state, lo=Vector(0, 0, 0),
                             hi=Vector(30, 30.0, 60))
state.atomParams.addSpecies(handle='substrate', mass=1)



ljcut = FixLJCut(state, handle='ljcut')


#deposited atom iteraction parameters
ljcut.setParameter(param='eps', handleA='substrate', handleB='substrate', val=1)
ljcut.setParameter(param='sig', handleA='substrate', handleB='substrate', val=.6)

bondHarm = FixBondHarmonic(state, 'bondharm')
angleHarm = FixAngleHarmonic(state, 'angleHarm')
dihedralOPLS = FixDihedralOPLS(state, 'opls')



state.activateFix(ljcut)
state.activateFix(bondHarm)
state.activateFix(angleHarm)
state.activateFix(dihedralOPLS)



#deposit substrate
substrateInitBounds = Bounds(state, lo=Vector(state.bounds.lo[0], state.bounds.lo[1], 2.5), hi=Vector(state.bounds.hi[0], state.bounds.hi[1], 7.5))
InitializeAtoms.populateRand(state, bounds=substrateInitBounds, handle='substrate', n=5000, distMin = 0.6)



unitMass = 12.011
unitEng = 0.066
unitLen = 3.500000
subTemp = 0.14
vaporTemp = 1.0
#InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments
state.createGroup('sub')
state.addToGroup('sub', [a.id for a in state.atoms])



def springFunc(id, pos):
    pos[2] = (substrateInitBounds.lo[2] + substrateInitBounds.hi[2]) / 2
    return pos

def springFuncEquiled(id, pos):
    return pos

fixSpring = FixSpringStatic(state, handle='substrateSpring', groupHandle='sub', k=10, tetherFunc=springFunc, multiplier=Vector(0.05, 0.05, 1))
state.activateFix(fixSpring)

fixNVT = FixNoseHoover(state, handle='nvt', groupHandle='sub', temp=subTemp, timeConstant=0.1)
state.activateFix(fixNVT)



integratorRelax = IntegratorRelax(state)
integratorRelax.set_params(dtMax_mult=1);
integratorRelax.run(15000, 1e-5)
print 'FINISHED FIRST RUN'
state.dt = 0.0005


fixSpring.tetherFunc = springFuncEquiled
fixSpring.updateTethers() #tethering to the positions they fell into
fixSpring.k = 1000
fixSpring.multiplier = Vector(1, 1, 1) #now spring holds in both dimensions

#InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments
#okay, substrate is set up, going to do deposition
integrator = IntegratorVerlet(state)


#REMEMBER CHARGES
ewald = FixChargeEwald(state, "chargeFix", "all")
ewald.setParameters(32, 1.0, 3)
state.activateFix(ewald)

reader = LAMMPS_Reader(state=state, unitLen = unitLen, unitMass = unitMass, unitEng = unitEng, bondFix = bondHarm, nonbondFix = ljcut,  angleFix = angleHarm, dihedralFix = dihedralOPLS, atomTypePrefix = 'EBZ_', setBounds=False)
reader.read(dataFn = 'ebz.data')

state.atomParams.setValues('substrate', atomicNum=1)
state.atomParams.setValues('EBZ_1', atomicNum=1)
state.atomParams.setValues('EBZ_2', atomicNum=1)
state.atomParams.setValues('EBZ_3', atomicNum=6)
state.atomParams.setValues('EBZ_4', atomicNum=6)

print("Average per particle energy: {}".format(integratorRelax.energyAverage()))

# Start deposition
toDeposit = [(7, 3), (6, 4)]
wallDist = 10
topWall = FixWallHarmonic(state, handle='wall', groupHandle='all', origin=Vector(0, 0, state.bounds.hi[2]), forceDir=Vector(0, 0, -1), dist=wallDist, k=100)

state.activateFix(topWall)

state.createMolecule([a.id for a in state.atoms if 'EBZ' in a.type])
newMolecs = [state.molecules[0]]
newMolecIds = list(newMolecs[0].ids)
newMolecPoses = [newMolecs[0].COM()]
firstRunDone = False

depositionRuns = 3
nMolecsPerDep = 15
for i in range(depositionRuns):

    while len(newMolecs) < nMolecsPerDep:
        state.duplicateMolecule(state.molecules[-1])
        newMolecs.append(state.molecules[-1])
        newMolecIds += [id for id in newMolecs[-1].ids]
    maxZ = max(a.pos[2] for a in state.atoms if not a.id in newMolecIds)
    #print [a.pos[2] for a in state.atoms]
    assert(maxZ < state.bounds.hi[2] - 2*wallDist)
    newZ = maxZ + 2.25
    for newMolec in newMolecs:
        COM = newMolec.COM()
        def tooClose():
            for pos in newMolecPoses:
                myPos = Vector(x, y, pos[2])
                if pos.dist(myPos) < 3:
                    return True
            else:
                return False

        x = state.bounds.lo[0] + (state.bounds.hi[0] - state.bounds.lo[0]) * random()
        y = state.bounds.lo[1] + (state.bounds.hi[1] - state.bounds.lo[1]) * random()
        while tooClose():
            x = state.bounds.lo[0] + (state.bounds.hi[0] - state.bounds.lo[0]) * random()
            y = state.bounds.lo[1] + (state.bounds.hi[1] - state.bounds.lo[1]) * random()
        newMolec.translate(Vector(x, y, newZ) - COM)
        newMolecPoses.append(newMolec.COM())
    vaporGrpName = 'vapor'
    state.createGroup(vaporGrpName, newMolecIds)

    state.addToGroup(vaporGrpName, newMolecIds)
    InitializeAtoms.initTemp(state, vaporGrpName, vaporTemp)
    state.destroyGroup(vaporGrpName)
    print newMolecIds
    for id in newMolecIds:
        a = state.atoms[state.idToIdx(id)]
        a.vel[2] -= 1

    #for a in state.atoms:
        #print a.vel

    writer = WriteConfig(state, handle='writer', fn='pvd_test_%d' % i, format='xyz', writeEvery=1000)
    writer.unitLen = 1/unitLen
    state.activateWriteConfig(writer)
    integrator.run(100000)
    state.deactivateWriteConfig(writer)
    for bond in bondHarm.bonds:
        idxA, idxB = state.idToIdx(bond.ids[0]), state.idToIdx(bond.ids[1])
    newMolecIds = []
    newMolecs = []
    newMolecPoses = []
#and we're done!


