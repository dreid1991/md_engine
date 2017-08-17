import sys
import matplotlib.pyplot as plt
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
sys.path.append('../util_py')
import water
from water import *
import math
from DASH import *
import numpy as np

state = State()
state.deviceManager.setDevice(0)

##############################
# Set initial density here
##############################
numMolecules = 512
sideLength = 27.0

loVector = Vector(0,0,0)
hiVector = Vector(sideLength, sideLength, sideLength)

state.bounds = Bounds(state, lo = loVector, hi = hiVector)
state.rCut = 10.0
state.padding = 1.0
state.periodicInterval = 7
state.shoutEvery = 100
state.units.setReal()

# real units --> 0.5 ~ half femtosecond
state.dt = 0.5

COMV_By_Turn = []
COM_By_Turn = []
def computeCOMV(currentTurn):
    COMV = Vector(0, 0, 0)
    COM  = Vector(0, 0, 0)
    massTotal = 0.0
    for atom in state.atoms:
        COMV += atom.vel * atom.mass
        COM  += atom.pos * atom.mass
        massTotal += atom.mass
    COM  *= (1.0 / massTotal)
    COMV *= (1.0 / massTotal)

    COMV_By_Turn.append(COMV)
    COM_By_Turn.append(COM)


COMV_Simulation = PythonOperation("COMV",5,computeCOMV)
state.activatePythonOperation(COMV_Simulation)

# handles for our atoms
oxygenHandle = 'OW'
hydrogenHandle = 'HY'
mSiteHandle = 'M'

# add our oxygen and hydrogen species to the simulation
state.atomParams.addSpecies(handle=oxygenHandle, mass=15.9994, atomicNum=8)
state.atomParams.addSpecies(handle=hydrogenHandle, mass=1.008, atomicNum=1)

# from TIP4P/2005 paper (Abascal & Vega , J. Chem. Phys. 123, 234505 (2005))

#epsilon = epsPerKb * kb * N_A * JtoKcal
#print "epsilon was found to be", epsilon
epsKjPerMol = 0.63627 #kJ per Mol
sigma = 3.15066
kjToKCal = 0.239006
epsilon = epsKjPerMol * kjToKCal


nonbond = FixLJCut(state,'cut')
nonbond.setParameter('sig',oxygenHandle, oxygenHandle, sigma)
nonbond.setParameter('eps',oxygenHandle, oxygenHandle, epsilon)
nonbond.setParameter('sig',hydrogenHandle, hydrogenHandle, 0.0)
nonbond.setParameter('eps',hydrogenHandle, hydrogenHandle, 0.0)
rigid = FixRigid(state,'rigid','all')

# our vector of centers
positions = []

xyzrange = int(math.ceil(numMolecules**(1.0/3.0)))
xyzFloat = float(xyzrange)
for x in xrange(xyzrange):
    for y in xrange(xyzrange):
        for z in xrange(xyzrange):
            pos = Vector( float(x)/(xyzFloat)*(0.9*sideLength) + 0.05*sideLength,
                          float(y)/(xyzFloat)*(0.9*sideLength) + 0.05*sideLength,
                          float(z)/(xyzFloat)*(0.9*sideLength) + 0.05*sideLength)
            positions.append(pos)

velocity = Vector(0,0,0)

for i in range(numMolecules):
    center = positions[i]
    # we are using real units, so skip bondLength argument
    molecule = create_TIP3P(state,oxygenHandle,hydrogenHandle,center=center,orientation="random")
    ids = []
    for atomId in molecule.ids:
        ids.append(atomId)


    rigid.createRigid(ids[0], ids[1], ids[2])
    for j in range(3):
        state.atoms[ids[j]].vel = velocity

#print 'done adding molecules to simulation'
#print 'distance between molecules 1 and 2: ', positions[1] - positions[0]
state.activateFix(rigid)
#InitializeAtoms.initTemp(state, 'all',298.15)
#fixNPT = FixNoseHoover(state,'npt','all')
#fixNPT.setTemperature(298.15,100.0*state.dt)
#fixNPT.setPressure('ANISO',1.0,1000*state.dt)
#state.activateFix(fixNPT)

# and then we have charges to take care of as well
#charge = FixChargeEwald(state, 'charge', 'all')
#charge.setParameters(64)
#state.activateFix(charge)
state.activateFix(nonbond)

integVerlet = IntegratorVerlet(state)

tempData = state.dataManager.recordTemperature('all','scalar', 1)
#tempData = state.dataManager.recordTemperature('all','scalar', 100)
pressureData = state.dataManager.recordPressure('all','scalar', 1)
engData = state.dataManager.recordEnergy('all','scalar',1)
boundsData = state.dataManager.recordBounds(100)

writeconfig = WriteConfig(state, fn='tip3p_out', writeEvery=5, format='xyz', handle='writer')
state.activateWriteConfig(writeconfig)

print 'about to run!'
integVerlet.run(1000)

for index, item in enumerate(COMV_By_Turn):
    print item, COM_By_Turn[index]


#print state.bounds.volume()
#print pressureData.vals
#print engData.vals
#print sumV / len(state.atoms)/3.0
#plt.plot(pressureData.turns, pressureData.vals)
#plt.show()
#plt.show()
#state.dataManager.stopRecord(tempData)
#integVerlet.run(10000)
#print len(tempData.vals)
#plt.plot([x for x in engData.vals])
#plt.show()
#print sum(tempData.vals) / len(tempData.vals)
#print boundsData.vals[0].getSide(1)
#print engData.turns[-1]
#print 'last eng %f' % engData.vals[-1]
#print state.turn
#print integVerlet.energyAverage('all')
#perParticle = integVerlet.energyPerParticle()
#print sum(perParticle) / len(perParticle)
