import sys
import os

# relative path to /build/ dir
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
sys.path.append('../util_py')

from DASH import *
from LAMMPS_Reader import LAMMPS_Reader
import argparse
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
state.rCut = 10.0
state.padding = 1.0
state.periodicInterval = 7
state.shoutEvery = 100
state.dt = 0.5


##############################################
# PIMD parameters - as from examples from Mike
##############################################
nBeads = 1;
the_temp = 300.0;
the_temp *= nBeads

# handles for our atoms
oxygenHandle = 'OW'
hydrogenHandle = 'HY'
mSiteHandle = 'M'

# add our oxygen and hydrogen species to the simulation
state.atomParams.addSpecies(handle=oxygenHandle, mass=15.9994, atomicNum=8)
state.atomParams.addSpecies(handle=hydrogenHandle, mass=1.008, atomicNum=1)
state.atomParams.addSpecies(handle=mSiteHandle,mass=0.00,atomicNum=0)

##############################################################
# q-TIP4P/F Parameters - from J. Chem. Phys. 131 024501 (2009)
##############################################################
epsilon = 0.1852 # kcal/mol
sigma = 3.1589 # angstroms


#####################
# Charge interactions
#####################
charge = FixChargeEwald(state, 'charge', 'all')
charge.setParameters(128,state.rCut-1, 3)

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
# The quartic bond potential, and harmonic angle potential
#############################################################
bondQuartic = FixBondQuartic(state,'bondQuart')
bondQuart.setBondTypeCoefs(type=0, k2=607.19, k3=-1388.65, k4=1852.58, r0=0.9419)

harmonicAngle = FixAngleHarmonic(state,'angleH')
harmonicAngle.setAngleTypeCoefs(type=0, k=87.85, theta0=( (107.4/180.0) * pi))
#############################################################

# our vector of centers
positions = []

# approximately on a lattice within the box to preclude overlapping molecules
# -- their random orientation will help relax the system quicker
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
    
    # add the atom ids to our instance of FixTIP4PFlexible
    flexibleTIP4P.addMolecule(ids[0], ids[1], ids[2], ids[3])
    
    # create the intramolecular OH bonds
    bondQuartic.createBond(state.atoms[ids[0]], state.atoms[ids[1]], type=0)
    bondQuartic.createBond(state.atoms[ids[0]], state.atoms[ids[2]], type=0)

    # and the harmonic angle HOH
    harmonicAngle.createAngle(state.atoms[ids[1]], state.atoms[ids[0]], state.atoms[ids[2]],type=0)

print 'done adding molecules to simulation'

#############################################################
# Initialization of potentials
#############################################################

##############################
# Intermolecular Interactions:
# - Charges
# - LJ
##############################
state.activateFix(charge)
state.activateFix(nonbond)

##############################
# Intramolecular Interactions
# -M-site distribution
# -HOH Angle
# -Quartic bonds
##############################
state.activateFix(flexibleTIP4P)
state.activateFix(harmonicAngle)
state.activateFix(bondQuartic)

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


#################################
# and some calls to PIMD
#################################
#state.nPerRingPoly = nBeads
#state.preparePIMD(the_temp)

writer = WriteConfig(state, handle='writer', fn='configPIMD', format='xyz',
                     writeEvery=1)
state.activateWriteConfig(writer)

integVerlet.run(5000)

fid = open('thermo.dat', "w")

for t, T in zip(tempData.turns, tempData.vals):
    fid.write("{:<8}{:>15.5f}\n".format(t,T))

