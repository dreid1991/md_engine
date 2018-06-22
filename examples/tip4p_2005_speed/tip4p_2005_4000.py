import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
# relative path to /build/ dir
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7']
sys.path.append('../../util_py')

from DASH import *
import water
from water import *
import math
import numpy as np

def read_gromacs_trajectory(filename,nFrames,hasPositions,hasVelocities,hasForces,getBox):
    positions = []
    velocities = []
    forces = []

    with open(filename) as f:
        for i in range(nFrames):
            f.readline()
            line = f.readline().split()
            nAtoms = int(line[1])
            f.readline()
            line = re.split('\s+|, ',f.readline())
            box_xdim = float(line[3])
            line = re.split('\s+|, ',f.readline())
            box_ydim = float(line[5])
            line = re.split('\s+|, |}',f.readline())
            box_zdim = float(line[7])
            box_xdim *= 10.0
            box_ydim *= 10.0
            box_zdim *= 10.0
            if (hasPositions):
                f.readline()
                for j in range(nAtoms):
                    x = re.split('{|,|}\s+',f.readline())
                    try:
                        xpos = float(x[1].lstrip().rstrip()) * 10.0
                        ypos = float(x[2].lstrip().rstrip()) * 10.0
                        zpos = float(x[3].lstrip().rstrip()) * 10.0
                        this_pos = np.asarray((xpos,ypos,zpos))
                        positions.append(this_pos)
                    except:
                        print 'Caught an exception at atom {}'.format(j)
                        print x[1], x[2], x[3]
                        exit()
            if (hasVelocities):
                f.readline()
                for j in range(nAtoms):
                    v = re.split('{|,|}\s+',f.readline())
                    xvel = float(v[1].lstrip().rstrip()) * 0.01
                    yvel = float(v[2].lstrip().rstrip()) * 0.01
                    zvel = float(v[3].lstrip().rstrip()) * 0.01
                    this_vel = np.asarray((xvel,yvel,zvel))
                    velocities.append(this_vel)

            if (hasForces):
                f.readline()
                for j in range(nAtoms):
                    pass
                    # TODO unit conversion for GMX forces to DASH forces..?
                    #f = re.split('{ |,|}\s+',f.readline())
                    #xf = float(x[1].strip()) * 0.01
                    #yf = float(x[2].strip()) * 0.01
                    #zf = float(x[3].strip()) * 0.01
                    #this_vel = np.asarray((xvel,yvel,zvel))
                    #velocities.append(this_vel)
    box = np.asarray((box_xdim,box_ydim,box_zdim))
    return positions,velocities,forces,box


state = State()
state.deviceManager.setDevice(0)

##############################
# Set initial density here
##############################
numMolecules = 4000
numTurnsProd = 200000

###############################
# Calculating box dimensions
###############################

#######################################################
# setting the box bounds and other attributes of state
#######################################################
#loVector = Vector(0,0,0)
#hiVector = Vector(sideLength, sideLength, sideLength)

state.units.setReal()

# grab the bounds from tip4p_init.out

positions,velocities,forces,box=read_gromacs_trajectory("tip4p_init_4000.out",1,True,False,False,True)


#for index, position in enumerate(positions):
#    print index, position
#
#for index, velocity in enumerate(velocities):
#    print index, velocity

print box

lo = Vector(0.0, 0.0, 0.0)
hi = lo + Vector(box[0], box[1], box[2])
state.bounds = Bounds(state, lo = lo, hi = hi)
# if rCut + padding > 0.5 sideLength, you will have a bad day
state.rCut = 10.0
state.padding = 1.0
state.periodicInterval = 7
state.shoutEvery = 500
state.dt = 1.0

# handles for our atoms
oxygenHandle = 'OW'
hydrogenHandle = 'HY'
mSiteHandle = 'OM'
# add our oxygen and hydrogen species to the simulation
state.atomParams.addSpecies(handle=oxygenHandle, mass=15.9994, atomicNum=8)
state.atomParams.addSpecies(handle=hydrogenHandle, mass=1.008, atomicNum=1)
state.atomParams.addSpecies(handle=mSiteHandle, mass=0.0, atomicNum=0)

##############################################################
# TIP3P parameters
##############################################################
epsilon = 0.185207422 # given in kcal/mol
sigma = 3.1589 # given in Angstroms

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
nonbond.setParameter('eps',mSiteHandle, mSiteHandle,0.0)
nonbond.setParameter('sig',mSiteHandle, mSiteHandle,0.0)


#############################################################
# Instantiate FixRigid
#############################################################
rigid = FixRigid(state,handle='rigid',style='TIP4P/2005')

# we don't really need to create the molecules.. but we do need to call 'add atom'
# for each atom in the molecule, and then createRigid.  We do need to add ghost particles
# explicitly.
# Note that

# XXX here we add our call to readig tip4p_init.out, setting the positions, velocities,
# and forces on each atom.  Then we will propagate the system.
# hopefully this doesn't explode, in which case we are good to go, and the
# only thing that was wrong was the initial conditions!



# create the molecules, and add them to the FixFlexibleTIP4P fix
for i in range(numMolecules):
    OPos = positions[i*4]
    OPos = Vector(OPos[0], OPos[1], OPos[2])

    H1Pos= positions[i*4 + 1]
    H1Pos= Vector(H1Pos[0], H1Pos[1], H1Pos[2])

    H2Pos= positions[i*4 + 2]
    H2Pos= Vector(H2Pos[0], H2Pos[1], H2Pos[2])

    MPos = positions[i*4 + 3]
    MPos = Vector(MPos[0], MPos[1], MPos[2])
    #molecule = create_TIP3P(state,oxygenHandle,hydrogenHandle,
    #        bondLength=0.9572,center=center,orientation="random")
    molecule = create_TIP4P_2005(state,oxygenHandle,hydrogenHandle,mSiteHandle,OPos=OPos,
                                 H1Pos=H1Pos,H2Pos=H2Pos,MPos=MPos)
    ids = []
    for atomId in molecule.ids:
        ids.append(atomId)

    rigid.createRigid(ids[0], ids[1], ids[2],ids[3])


hasVelocities = False
if (hasVelocities):
    for atom in state.atoms:
        thisId = atom.id
        vel = velocities[thisId]
        velAsVector = Vector(vel[0], vel[1], vel[2])
        atom.vel = velAsVector
        # and we'll calculate our own forces..
print 'done adding molecules to simulation'

#############################################################
# Initialization of potentials
#############################################################
InitializeAtoms.initTemp(state, 'all', 320.0)

#####################
# Charge interactions
#####################
charge = FixChargeEwald(state, 'charge', 'all')
charge.setParameters(64,-1,5)
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


###########################################################
# NVE Dynamics
##########################################################
# write every 10 fs
#print '\nnow doing NVE run!\n'
integVerlet.run(numTurnsProd)

###########################################################
# Output for analysis
###########################################################

print "\nSimulation has concluded.\n"

