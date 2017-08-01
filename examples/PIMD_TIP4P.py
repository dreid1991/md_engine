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
numMolecules = 512
sideLength = 27.0

loVector = Vector(0,0,0)
hiVector = Vector(sideLength, sideLength, sideLength)

state.units.setReal()

state.bounds = Bounds(state, lo = loVector, hi = hiVector)
state.rCut = 9.0
state.padding = 1.0
state.periodicInterval = 7
state.shoutEvery = 1000
state.dt = 1.0


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
charge.setParameters(128,state.rCut-1, 3)
state.activateFix(charge)


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
flexibleTIP4P = FixTIP4PFlexible(state,'TIP4PFlexible', 'all')
# and all that remains is to add each molecule to TIP4PFlexible

#############################################################
# and also, for now an arbitrary harmonic bond potential
# just so things don't explode.
#############################################################
#harmonicBonds = FixBondHarmonic(state,'harmonic')
#harmonicAngle = FixAngleHarmonic(state,'angleH')

#state.activateFix(harmonicBonds)
#state.activateFix(harmonicAngle)

anglePot = FixAngleHarmonic(state, 'angleHarm')
anglePot.setAngleTypeCoefs(type=0, k=87.85, theta0=(107.4/180.0)*pi)

bondQuart = FixBondQuartic(state, 'bondQuart')
bondQuart.setBondTypeCoefs(type=0, k2=607.19,k3=-1388.65,k4=1852.58,r0=0.9419)



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

    # add quartic bonds so that the molecules don't collapse
    # and angle harmonic angle 0 1 2
    bondQuart.createBond(state.atoms[ids[0]], state.atoms[ids[1]], type=0)
    bondQuart.createBond(state.atoms[ids[0]], state.atoms[ids[2]], type=0)
    anglePot.createAngle(state.atoms[ids[1]], state.atoms[ids[0]], state.atoms[ids[2]], type=0)



    # make a harmonic OH1 bond with some high stiffness, and OH2, and H1H2
    '''
    harmonicBonds.createBond(state.atoms[ids[0]], state.atoms[ids[1]],
                             k=200000, r0 = 0.9419)
    harmonicBonds.createBond(state.atoms[ids[0]], state.atoms[ids[2]],
                             k=200000, r0 = 0.9419)
    harmonicAngle.createAngle(state.atoms[ids[1]], state.atoms[ids[0]], state.atoms[ids[2]],
                              k=200000, theta0=1.8744836)
    '''
print 'done adding molecules to simulation'
#############################################################
# Initialization of potentials
#############################################################

#####################
# LJ Interactions
#####################
# -- we defined the LJ interactions above
state.activateFix(nonbond)
#####################
# Intramolecular Bonds - Quartic and Angle potentials
#####################
state.activateFix(bondQuart)
state.activateFix(anglePot)

########################################################################################
# Flexible TIP4P - note that this /only/ distributes the M-site forces!
# Other components of the potential must be assembled using the other fixes available.
########################################################################################
state.activateFix(flexibleTIP4P)

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
tempData = state.dataManager.recordTemperature('all', interval = 1)

################################################
# Thermostatting
################################################
fixNVT = FixNoseHoover(state,'nvt','all')
fixNVT.setTemperature(330.0, 200*state.dt)
state.activateFix(fixNVT)

tempData = state.dataManager.recordTemperature('all','vector', interval = 1)
enerData = state.dataManager.recordEnergy('all', 'scalar', interval = 1, fixes = [nonbond,charge,anglePot,bondQuart] )

#################################
# and some calls to PIMD
#################################
#state.nPerRingPoly = nBeads
#state.preparePIMD(the_temp)

writer = WriteConfig(state, handle='writer', fn='configPIMD', format='xyz',
                     writeEvery=10)
state.activateWriteConfig(writer)

integVerlet.run(10000)

print 'deactivating thermostat\n'

for index, item in enumerate(enerData.vals):
    print enerData.vals[index], sum(tempData.vals[index]), enerData.vals[index] + sum(tempData.vals[index])

state.deactivateFix(fixNVT)

integVerlet.run(50000)

for index, item in enumerate(enerData.vals):
    print enerData.vals[index], sum(tempData.vals[index]), enerData.vals[index] + sum(tempData.vals[index])

state.dt = 1.0
fid = open('thermo.dat', "w")

for t, T in zip(tempData.turns, tempData.vals):
    fid.write("{:<8}{:>15.5f}\n".format(t,T))

