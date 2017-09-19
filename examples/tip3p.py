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
numMolecules = 1000
density = 1.002
# consider dt = 1.0; then, 5 ps rescale
numTurnsEquil_Rescale = 5000
# do 195 ps NVT
numTurnsEquil_NVT = 95000
# do 800 ps NVE
numTurnsProd = 100000

###############################
# Calculating box dimensions
###############################
gPerMol = 18.0154
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
state.shoutEvery = 500
state.dt = 1.0

print "running for %d equil turns and %d prod turns with a timestep of %f" %(numTurnsEquil_Rescale + numTurnsEquil_NVT, numTurnsProd, state.dt)
print "there are %d molecules in this simulation" %numMolecules

# handles for our atoms
oxygenHandle = 'OW'
hydrogenHandle = 'HY'

# add our oxygen and hydrogen species to the simulation
state.atomParams.addSpecies(handle=oxygenHandle, mass=15.9994, atomicNum=8)
state.atomParams.addSpecies(handle=hydrogenHandle, mass=1.008, atomicNum=1)

##############################################################
# TIP3P parameters
##############################################################
epsilon = 0.15207 # given in kcal/mol
sigma = 3.15066 # given in Angstroms


if (sigma != 3.15066):
    print "WRONG SIGMA VALUE FOR TIP3P IN SCRIPT."
    exit(1)

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
rigid = FixRigid(state,'rigid','all')

# our vector of centers
positions = []

xyzrange = int(math.ceil(numMolecules**(1.0/3.0)))
xyzFloat = float(xyzrange)
for x in xrange(xyzrange):
    for y in xrange(xyzrange):
        for z in xrange(xyzrange):
            pos = Vector( float(x)/(xyzFloat)*(0.98*sideLength),
                          float(y)/(xyzFloat)*(0.98*sideLength),
                          float(z)/(xyzFloat)*(0.98*sideLength))

            #print pos
            positions.append(pos)


# create the molecules, and add them to the FixFlexibleTIP4P fix
for i in range(numMolecules):
    center = positions[i]

    molecule = create_TIP3P(state,oxygenHandle,hydrogenHandle,
            bondLength=0.9572,center=center,orientation="random")
    ids = []
    for atomId in molecule.ids:
        ids.append(atomId)

    rigid.createRigid(ids[0], ids[1], ids[2])


print 'done adding molecules to simulation'

#############################################################
# Initialization of potentials
#############################################################

#####################
# Charge interactions
#####################
charge = FixChargeEwald(state, 'charge', 'all')
charge.setError(0.01,state.rCut-1, 3)
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
InitializeAtoms.initTemp(state, 'all', 350.0)

########################################
# our integrator
########################################
integVerlet = IntegratorVerlet(state)

########################################
# Data recording
########################################
# Record the scalar temperature every 10 steps
# Record the total potential energy every 10 steps
# Record the center of mass velocity every 10 steps

tempData = state.dataManager.recordTemperature('all','scalar', interval = 10)
enerData = state.dataManager.recordEnergy('all', 'scalar', interval = 10,fixes=[charge,nonbond])
comvData = state.dataManager.recordCOMV(interval=10)

# write every 10 fs
writer = WriteConfig(state, handle='writer', fn='rigid_tip3p', format='xyz',
                     writeEvery=10)
state.activateWriteConfig(writer)

###########################################################
# Isokinetic velocity rescaling: initial equilibration
###########################################################
fixNVT = FixNVTRescale(state,'nvt','all',298.00,50)
state.activateFix(fixNVT)
integVerlet.run(numTurnsEquil_Rescale)
state.deactivateFix(fixNVT)

###########################################################
# Nose-Hoover NVT Dynamics: further equilibration
###########################################################
NVT = FixNoseHoover(state,'nvt','all')
NVT.setTemperature(298.15, 100*state.dt)
state.activateFix(NVT)
integVerlet.run(numTurnsEquil_NVT)
state.deactivateFix(NVT)

##########################################################
# NVE Dynamics
##########################################################
print '\nnow doing NVE run!\n'
integVerlet.run(numTurnsProd)

###########################################################
# Output for analysis
###########################################################


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
