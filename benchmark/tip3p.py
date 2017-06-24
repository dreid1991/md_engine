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
state.deviceManager.setDevice(1)


# VERY IMPORTANT: the initial density of our simulation cell
numMolecules = 200
sideLength = 45.0



loVector = Vector(0,0,0)
hiVector = Vector(sideLength, sideLength, sideLength)
state.bounds = Bounds(state, lo = loVector, hi = hiVector)
state.rCut = 5.0
state.padding = 1.0
state.periodicInterval = 7
state.shoutEvery = 100
state.units.setReal()
state.dt = 0.5
upperBound = hiVector[0]

# handles for our atoms
oxygenHandle = 'OW'
hydrogenHandle = 'HY'
mSiteHandle = 'M'

# add our oxygen and hydrogen species to the simulation
state.atomParams.addSpecies(handle=oxygenHandle, mass=15.9994, atomicNum=8)
state.atomParams.addSpecies(handle=hydrogenHandle, mass=1.008, atomicNum=1)
state.atomParams.addSpecies(handle=mSiteHandle,mass=0,atomicNum=0)

# equilibrate NPT 298k, 1 bar for 1m steps
# then run NVE for 5m steps for computing the O-O RDF
# -- during this time, compute t_average and p_average

# from TIP4P/2005 paper (Abascal & Vega , J. Chem. Phys. 123, 234505 (2005))
epsPerKb = 93.2
sigma = 3.1589

# kb and N_A, without e+23 and e-23 (cancels out on multiplication)
kb = 1.38064852; #e-23
N_A = 6.0221409; #e+23
JtoKcal = 1.0 / 4184.0

epsilon = epsPerKb * kb * N_A * JtoKcal
#print "epsilon was found to be", epsilon

# convert the TIP4P epsilon from 93.2 [J/K] to kcal/mol
nonbond = FixLJCut(state,'cut')
nonbond.setParameter('sig',oxygenHandle, oxygenHandle, sigma)
nonbond.setParameter('eps',oxygenHandle, oxygenHandle, epsilon)

rigid = FixRigid(state,'rigid','all')

# our vector of centers
positions = []

xyzrange = int(math.ceil(numMolecules**(1.0/3)))
xyzFloat = float(xyzrange)
for x in xrange(xyzrange):
    for y in xrange(xyzrange):
        for z in xrange(xyzrange):
            pos = Vector( float(x)/(xyzFloat)*sideLength,
                          float(y)/(xyzFloat)*sideLength,
                          float(z)/(xyzFloat)*sideLength)
            #print pos
            positions.append(pos)

velocity = Vector(0.0, 0.0, 0.0)
for i in range(numMolecules):
    center = positions[i]
    molecule = create_TIP3P(state,oxygenHandle,hydrogenHandle,center)
    ids = []
    for atomId in molecule.ids:
        state.atoms[atomId].vel = velocity
        ids.append(atomId)


    rigid.createRigid(ids[0], ids[1], ids[2])

print 'done adding molecules to simulation'
print 'distance between molecules 1 and 2: ', positions[1] - positions[0]
state.activateFix(rigid)
#InitializeAtoms.initTemp(state, 'all',298.15)
#fixNPT = FixNoseHoover(state,'npt','all')
#fixNPT.setTemperature(298.15,100.0*state.dt)
#fixNPT.setPressure('ANISO',1.0,1000*state.dt)
#state.activateFix(fixNPT)

# and then we have charges to take care of as well
charge = FixChargeEwald(state, 'charge', 'all')
charge.setParameters(64)
state.activateFix(charge)

integVerlet = IntegratorVerlet(state)

tempData = state.dataManager.recordTemperature('all','scalar', 1)
#tempData = state.dataManager.recordTemperature('all','scalar', 100)
pressureData = state.dataManager.recordPressure('all','scalar', 1)
#engData = state.dataManager.recordEnergy('all', 100)
boundsData = state.dataManager.recordBounds(100)

writeconfig = WriteConfig(state, fn='tip3p_out', writeEvery=1, format='xyz', handle='writer')
state.activateWriteConfig(writeconfig)

print 'about to run!'
integVerlet.run(1000)
sumV = 0.
for a in state.atoms:
    sumV += a.vel.lenSqr()
print state.bounds.volume()
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
