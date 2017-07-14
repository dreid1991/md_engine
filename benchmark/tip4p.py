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
numMolecules = 2056
sideLength = 50.0

loVector = Vector(0,0,0)
hiVector = Vector(sideLength, sideLength, sideLength)

state.bounds = Bounds(state, lo = loVector, hi = hiVector)
state.rCut = 9.0
state.padding = 1.0
state.periodicInterval = 7
state.shoutEvery = 100
state.units.setReal()
state.dt = 0.1

# handles for our atoms
oxygenHandle = 'OW'
hydrogenHandle = 'HY'
mSiteHandle = 'M'

# add our oxygen and hydrogen species to the simulation
state.atomParams.addSpecies(handle=oxygenHandle, mass=15.9994, atomicNum=8)
state.atomParams.addSpecies(handle=hydrogenHandle, mass=1.008, atomicNum=1)
state.atomParams.addSpecies(handle=mSiteHandle,mass=0.00,atomicNum=0)

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
nonbond.setParameter('sig',hydrogenHandle, hydrogenHandle, 0.0)
nonbond.setParameter('eps',hydrogenHandle, hydrogenHandle, 0.0)
nonbond.setParameter('sig',mSiteHandle, mSiteHandle, 0.0)
nonbond.setParameter('eps',mSiteHandle, mSiteHandle, 0.0)


# and then we have charges to take care of as well
#charge = FixChargeEwald(state, 'charge', 'all')
#charge.setParameters(128)

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

            #print pos
            positions.append(pos)

e3b3 = FixE3B3(state,'e3b3','all')

for i in range(numMolecules):
    center = positions[i]
    molecule = create_TIP4P_2005(state,oxygenHandle,hydrogenHandle,mSiteHandle,center,"random")

    ids = []
    for atomId in molecule.ids:
        ids.append(atomId)

    state.atoms[ids[-1]].mass = 0.0
    rigid.createRigid(ids[0], ids[1], ids[2], ids[3])
    e3b3.addMolecule(ids[0], ids[1], ids[2], ids[3])

print 'done adding molecules to simulation'
InitializeAtoms.initTemp(state, 'all',300.0)

#############################################################
# Initialization of potentials
#############################################################

#####################
# Charge interactions
#####################
charge = FixChargeEwald(state, 'charge', 'all')
charge.setParameters(128)
state.activateFix(charge)

#####################
# LJ Interactions
#####################
# -- we defined the LJ interactions above
state.activateFix(nonbond)

#####################
# Rigid Constraints
#####################
# - we created this prior to adding the atoms to the box
#rigid.printing = True
state.activateFix(rigid)

#####################
# E3B3
#####################
state.activateFix(e3b3)

#fixNPT = FixNoseHoover(state,'npt','all')
#fixNPT.setTemperature(298.15,100.0*state.dt)
#fixNPT.setPressure('ANISO',1.0,1000*state.dt)
#state.activateFix(fixNPT)

#state.activateFix(e3b3)
integVerlet = IntegratorVerlet(state)

tempData = state.dataManager.recordTemperature('all','scalar', 1)
#tempData = state.dataManager.recordTemperature('all','scalar', 100)
#pressureData = state.dataManager.recordPressure('all','scalar', 1)
#engData = state.dataManager.recordEnergy('all', 100)
boundsData = state.dataManager.recordBounds(100)

writeconfig = WriteConfig(state, fn='test_out', writeEvery=2, format='xyz', handle='writer')
state.activateWriteConfig(writeconfig)
integVerlet.run(500)

#fixNPT.setPressure('ANISO',1.0,1000*state.dt)
#integVerlet.run(50000)

sumV = 0.
for a in state.atoms:
    sumV += a.vel.lenSqr()
for index, item in enumerate(boundsData.vals):
    print item.volume()
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
print sum(tempData.vals) / len(tempData.vals)
#print boundsData.vals[0].getSide(1)
#print engData.turns[-1]
#print 'last eng %f' % engData.vals[-1]
#print state.turn
#print integVerlet.energyAverage('all')
#perParticle = integVerlet.energyPerParticle()
#print sum(perParticle) / len(perParticle)
