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

###############################################
# From q-TIP4P/F paper:
# "... all liquid simulations performed at a
#  temperature of 298.15K and density of
#  0.997 g/cm^-3 with 216 water molecules
#  in a cubic simulation box
###############################################
numMolecules = 1
sideLength = 18.6460776727

loVector = Vector(0,0,0)
hiVector = Vector(sideLength, sideLength, sideLength)

state.units.setReal()

state.bounds = Bounds(state, lo = loVector, hi = hiVector)
state.rCut = 9.0
state.padding = 0.5
state.periodicInterval = 5
state.shoutEvery = 100
state.dt = 0.500


##############################################
# PIMD parameters - as from examples from Mike
##############################################
nBeads = 32
the_temp = 298.15
the_temp *= nBeads

# handles for our atoms
oxygenHandle = 'OW'
hydrogenHandle = 'HY'
mSiteHandle = 'M'

# add our oxygen and hydrogen species to the simulation
state.atomParams.addSpecies(handle=oxygenHandle, mass=15.9994, atomicNum=8)
state.atomParams.addSpecies(handle=hydrogenHandle, mass=1.008, atomicNum=1)
state.atomParams.addSpecies(handle=mSiteHandle,mass=0.0,atomicNum=0)

##############################################################
# q-TIP4P/F Parameters - from J. Chem. Phys. 131 024501 (2009)
##############################################################
epsilon = 0.1852 # given in kcal/mol
sigma = 3.1589 # given in Angstroms

#####################
# Charge interactions
#####################
charge = FixChargeEwald(state, 'charge', 'all')
charge.setError(0.01,state.rCut,3)


nonbond = FixLJCut(state,'cut')
nonbond.setParameter('sig',oxygenHandle, oxygenHandle, sigma)
nonbond.setParameter('eps',oxygenHandle, oxygenHandle, epsilon)
nonbond.setParameter('sig',hydrogenHandle, hydrogenHandle, 0.0)
nonbond.setParameter('eps',hydrogenHandle, hydrogenHandle, 0.0)
nonbond.setParameter('sig',mSiteHandle, mSiteHandle, 0.0)
nonbond.setParameter('eps',mSiteHandle, mSiteHandle, 0.0)

###########################################################
# Instantiate FixTIP4PFlexible
###########################################################
flexibleTIP4P = FixTIP4PFlexible(state,'TIP4PFlexible')
# and all that remains is to add each molecule to TIP4PFlexible

#############################################################
# Instantiate the quartic bond (intramolecular OH) and
# the HOH angle potential
#############################################################
bondQuart = FixBondQuartic(state,'bondQuart')
bondQuart.setBondTypeCoefs(type=0,k2=607.19,k3=-1388.65,k4=1852.58,r0=0.9419)

harmonicAngle = FixAngleHarmonic(state,'angleH')
harmonicAngle.setAngleTypeCoefs(type=0, k=87.85, theta0=( (107.4/180.0) * pi))

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
    molecule = create_TIP4P_Flexible(state,oxygenHandle,hydrogenHandle,mSiteHandle,center,"random")

    ids = []
    for atomId in molecule.ids:
        ids.append(atomId)

    # sanity check: set the mass of the M-site to zero, again
    state.atoms[ids[-1]].mass = 0.00
    # add the atom ids to our instance of FixTIP4PFlexible
    flexibleTIP4P.addMolecule(ids[0], ids[1], ids[2], ids[3])

    #for atom in ids:
    #    print 'atom ', atom, ' with mass ', state.atoms[atom].mass, ' and force ', state.atoms[atom].force, ' placed at position ', state.atoms[atom].pos,'\n'

    bondQuart.createBond(state.atoms[ids[0]], state.atoms[ids[1]],type=0)
    bondQuart.createBond(state.atoms[ids[0]], state.atoms[ids[2]],type=0)

    harmonicAngle.createAngle(state.atoms[ids[1]], state.atoms[ids[0]], state.atoms[ids[2]],type=0)


print 'done adding molecules to simulation'
#############################################################
# Initialization of potentials
#############################################################

############################################
# Intermolecular interactions: LJ & Charge
############################################
# -- we defined the LJ interactions above
state.activateFix(nonbond)
state.activateFix(charge)

################################################################
# Intramolecular interactions:
# FixTIP4PFlexible, quartic bonds, and harmonic angle potential
################################################################
state.activateFix(bondQuart)
state.activateFix(flexibleTIP4P)
state.activateFix(harmonicAngle)

#############################################
# initialize at some temperature
#############################################
#InitializeAtoms.initTemp(state, 'all', 1.0)

########################################
# our integrator
########################################
integVerlet = IntegratorVerlet(state)

########################################
# Data recording
########################################
tempData = state.dataManager.recordTemperature('all', interval = 1)

################################################
# Thermostatting
################################################
fixNVT = FixNVTAndersen(state,'nvt','all',the_temp,0.5,5)
#fixNVT.setTemperature(300.0, 200*state.dt)
state.activateFix(fixNVT)

#fixNVT_Iso = FixNVTRescale(state,'nvt_iso','all',298.15,50)

#################################
# and some calls to PIMD
#################################
state.nPerRingPoly = nBeads
state.preparePIMD(the_temp)

writer = WriteConfig(state, handle='writer', fn='configPIMD', format='xyz',
                     writeEvery=1)
state.activateWriteConfig(writer)

writeRestart = WriteConfig(state, handle = 'restart',fn="tip4p_restart*", format='xml',writeEvery=1000)
state.activateWriteConfig(writeRestart)

writer.write();

integVerlet.run(500)


