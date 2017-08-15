import sys
import os
import matplotlib.pyplot as plt

# relative path to /build/ dir
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
sys.path.append('../util_py')

from DASH import *
import water
from water import *
import math
import numpy as np

state = State()
state.deviceManager.setDevice(0)

##############################
# Set initial density here
##############################
numMolecules = 1028
density = 1.0
# consider dt = 1.0; then, 5 ps rescale
numTurnsEquil_Rescale = 5000
# do 195 ps NVT
numTurnsEquil_NVT = 195000
# do 800 ps NVE
numTurnsProd = 800000

###############################
# Calculating box dimensions
###############################
gPerMol = 18.02
molPerG = 1.0 / gPerMol
cmPerAngstrom = 100 * (1.0 * (10 ** (-10.0)))
avogadro = 6.0221409e23
numberDensity = density * (cmPerAngstrom ** (3.0)) * molPerG * avogadro
sideLength = (numMolecules / numberDensity) ** (1.0/3.0)

print 'sideLength calculated to be: ', sideLength

#######################################################
# setting the box bounds and other attributes of state
#######################################################
loVector = Vector(0,0,0)
hiVector = Vector(sideLength, sideLength, sideLength)

state.units.setReal()

state.bounds = Bounds(state, lo = loVector, hi = hiVector)
# if rCut + padding > 0.5 sideLength, you will have a bad day
state.rCut = min(12.0, 0.45 * sideLength)
print 'rcut found to be ', state.rCut

state.padding = 1.0
state.periodicInterval = 7
state.shoutEvery = 5000
state.dt = 1.0

print "running for %d equil turns and %d prod turns with a timestep of %f" %(numTurnsEquil_Rescale + numTurnsEquil_NVT, numTurnsProd, state.dt)
print "there are %d molecules in this simulation" %numMolecules
##############################################
# PIMD parameters - as from examples from Mike
##############################################
nBeads = 32;
the_temp = 300.0;
the_temp *= nBeads

# handles for our atoms
oxygenHandle = 'OW'
hydrogenHandle = 'HY'
mSiteHandle = 'M'

# add our oxygen and hydrogen species to the simulation
state.atomParams.addSpecies(handle=oxygenHandle, mass=15.9994, atomicNum=8)
state.atomParams.addSpecies(handle=hydrogenHandle, mass=1.008, atomicNum=1)

##############################################################
# TIP3P parameters
##############################################################
epsilon = 0.1521 # given in kcal/mol
sigma = 3.5365 # given in Angstroms

######################
# LJ Interactions
######################
nonbond = FixLJCut(state,'cut')
nonbond.setParameter('sig',oxygenHandle, oxygenHandle, sigma)
nonbond.setParameter('eps',oxygenHandle, oxygenHandle, epsilon)
nonbond.setParameter('sig',hydrogenHandle, hydrogenHandle, 0.0)
nonbond.setParameter('eps',hydrogenHandle, hydrogenHandle, 0.0)
nonbond.setParameter('sig',oxygenHandle, hydrogenHandle, 0.0)
nonbond.setParameter('eps',oxygenHandle, hydrogenHandle, 0.0)


#############################################################
# and also, for now an arbitrary harmonic bond potential
# just so things don't explode.
#############################################################
#harmonicBonds = FixBondHarmonic(state,'harmonic')
#harmonicAngle = FixAngleHarmonic(state,'angleH')

#state.activateFix(harmonicBonds)
#state.activateFix(harmonicAngle)

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


# create the molecules, and add them to the FixFlexibleTIP4P fix
for i in range(numMolecules):
    center = positions[i]

    ## for TIP4P
   # molecule = create_TIP3P(state,oxygenHandle,hydrogenHandle,mSiteHandle,center,"random")

    molecule = create_TIP3P(state,oxygenHandle,hydrogenHandle,bondLength=0.9572,center=center,orientation="random")
    ids = []
    for atomId in molecule.ids:
        #state.atoms[atomId].vel = velocities
        #state.atoms[atomId].force= Vector(0,0,0)
        ids.append(atomId)


    # make a harmonic OH1 bond with some high stiffness, and OH2, and H1H2
    '''
    harmonicBonds.createBond(state.atoms[ids[0]], state.atoms[ids[1]],
                             k=200, r0 = 0.9572)
    harmonicBonds.createBond(state.atoms[ids[0]], state.atoms[ids[2]],
                             k=200, r0 = 0.9572)
    harmonicAngle.createAngle(state.atoms[ids[1]], state.atoms[ids[0]], state.atoms[ids[2]],
                              k=200, theta0=1.82421813)
    '''
    rigid.createRigid(ids[0], ids[1], ids[2])


print 'done adding molecules to simulation'

#############################################################
# Initialization of potentials
#############################################################

#####################
# Charge interactions
#####################
charge = FixChargeEwald(state, 'charge', 'all')
charge.setParameters(256,state.rCut-1, 3)
state.activateFix(charge)

#####################
# LJ Interactions
#####################
# -- we defined the LJ interactions above
state.activateFix(nonbond)

#############################################
# Rigid Fix
#############################################
state.activateFix(rigid)

#############################################
# initialize at some temperature
#############################################
InitializeAtoms.initTemp(state, 'all', 250.0)

#############################################
# Temperature control
#############################################
#fixNVT = FixNoseHoover(state,'nvt','all')
#fixNVT.setTemperature(300.0, 100*state.dt)
#state.activateFix(fixNVT)

fixNVT = FixNVTRescale(state,'nvt','all',298.15,100)
state.activateFix(fixNVT)
########################################
# our integrator
########################################
integVerlet = IntegratorVerlet(state)

########################################
# Data recording
########################################
#tempData = state.dataManager.recordTemperature('all','vector', interval = 1)
#enerData = state.dataManager.recordEnergy('all', interval = 1)

tempData = state.dataManager.recordTemperature('all','scalar', interval = 10)
enerData = state.dataManager.recordEnergy('all', 'scalar', interval = 10,fixes=[charge,nonbond])
comvData = state.dataManager.recordCOMV(interval=1)

#################################
# and some calls to PIMD
#################################
#state.nPerRingPoly = nBeads
#state.preparePIMD(the_temp)

# write every 10 fs
writer = WriteConfig(state, handle='writer', fn='rigid_tip3p', format='xyz',
                     writeEvery=10)
state.activateWriteConfig(writer)

integVerlet.run(numTurnsEquil_Rescale)
#print dir(tempData)
#print sum(tempData.vals[0])
state.deactivateFix(fixNVT)

NVT = FixNoseHoover(state,'nvt','all')
NVT.setTemperature(298.15, 100*state.dt)
state.activateFix(NVT)

# make new data computers



integVerlet.run(numTurnsEquil_NVT)

state.deactivateFix(NVT)

print '\nnow doing NVE run!\n'

integVerlet.run(numTurnsProd)

energyFile = open('Energy.dat','w')
for index, item in enumerate(enerData.vals):
    pe = str(enerData.vals[index])
    ke = str(tempData.vals[index])
    hamiltonian = str(enerData.vals[index] + tempData.vals[index])
    spacing = "    "
    energyFile.write(pe + spacing + ke + spacing + hamiltonian + "\n")

f = open('COMV.dat','w')
for index, item in enumerate(comvData.vals):
    xCOMV = str(item[0])
    yCOMV = str(item[1])
    zCOMV = str(item[2])
    mass = str(item[3])
    spacing = "   "
    f.write(xCOMV + spacing + yCOMV + spacing + zCOMV + spacing + mass + "\n")

print "\nSimulation has concluded.\n"
