import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7' ]
print sys.path
sys.path.append('../util_py')
from Sim import *
from math import *
import random
state = State()
state.deviceManager.setDevice(1)
state.bounds = Bounds(state, lo = Vector(-0, -0, -0), hi = Vector(45, 45, 45))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000
state.dt = 0.005

state.atomParams.addSpecies('spc1', 1, atomicNum=8)
state.atomParams.addSpecies('spc2', 1, atomicNum=1)
state.atomParams.addSpecies('spc3', 1, atomicNum=0)

sig = 3.15061
eps = 0.6364 # kJ/mol 
# length of OH bond / sig
offset1 = Vector(0.240255,-0.1859558,0)
offset2 = Vector(0.240255,0.1859558,0)
offset3 = Vector(0,0.15/sig,0)

tempUnit = 1.38e-23/(.6364*1000/6.022*10e23)
tSim = 300 * tempUnit 
# temp = ?, pressure = ?
# real temp in units K, Boltzmann constant J/K, eps should be in J
# pressure = sig^3/eps
# length of OH bond / sig                                                                                                                                                                                
offset1 = Vector(0.240255,-0.1859558,0)
offset2 = Vector(0.240255,0.1859558,0)

sig = 3.15061
eps = 0.6364 # kJ/mol                                                                                                                                                                                    
tempUnit = 1.38e-23/(.6364*1000/6.022*10e23)
tSim = 300 * tempUnit

# real temp in units K, Boltzmann constant J/K, eps should be in J                                                                                                                                      
# pressure = sig^3/eps                                                                                                                                                                                   
temp = tSim
sigSI = sig*1e-10
epsSI = eps*1000 / 6.022e23
pUnit = sigSI**3 / epsSI

# parameters for OH and HH?
nonbond = FixLJCut(state, 'cut')
nonbond.setParameter('sig', 'spc1', 'spc1', 1)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)

nonbond.setParameter('sig', 'spc1', 'spc2', 0)
nonbond.setParameter('eps', 'spc1', 'spc2', 0)

nonbond.setParameter('sig', 'spc2', 'spc2', 0)
nonbond.setParameter('eps', 'spc2', 'spc2', 0)

nonbond.setParameter('sig', 'spc3', 'spc3', 0)
nonbond.setParameter('eps', 'spc3', 'spc3', 0)

nonbond.setParameter('sig', 'spc3', 'spc1', 0)
nonbond.setParameter('eps', 'spc3', 'spc1', 0)

nonbond.setParameter('sig', 'spc3', 'spc2', 0)
nonbond.setParameter('eps', 'spc3', 'spc2', 0)

state.activateFix(nonbond)

positions = []
for x in xrange(28):
    for y in xrange(10):
        for z in xrange(10):
            pos = Vector(x*2+1,y*2+1,z*2+1)
            positions.append(pos)

# initialize rigid fix
rigid = FixRigid(state, 'rigid', 'all')

for i in xrange(2):
    position = positions[i]
    atomO = position
    atomH1 = position + offset1
    atomH2 = position + offset2
    atomM = position + offset3
    state.addAtom('spc1', atomO)
    state.addAtom('spc2', atomH1)
    state.addAtom('spc2', atomH2)
    state.addAtom('spc3', atomM)
    # normalize - divide by O
    massO = 15.9994
    massH = 1.00794
    state.atoms[i*4].mass = 1
    state.atoms[i*4+1].mass = massH/massO
    state.atoms[i*4+2].mass = massH/massO
    state.atoms[i*4+3].mass = 0
    state.atoms[i*4].q = -0.834
    state.atoms[i*4+1].q = 0.417
    state.atoms[i*4+2].q = 0.417
    state.atoms[i*4+3].q = -1.04
    # velocity
    velocity = Vector(0,0,0)
    for j in range(3):
        state.atoms[i*4 + j].vel = velocity
    #rigid.createRigid(i*3,i*3+2,i*3+1)
    rigid.createRigidTIP4P(i*4,i*4+2,i*4+1,i*4+3)

# need correct parameters for fixes

charge = FixChargeEwald(state, 'charge', 'all')
charge.setParameters(64, 3.0, 1)
state.activateFix(charge)

barostat = FixPressureBerendsen(state, 'barostat', 101325*pUnit, 1000*state.dt, 1)
state.activateFix(barostat)

nvt = FixNoseHoover(state, handle='nvt', groupHandle='all', temp=tSim, timeConstant=1000*state.dt)
state.activateFix(nvt)
InitializeAtoms.initTemp(state,'all',tSim)

state.activateFix(rigid)

#writeconfig = WriteConfig(state, fn='system', writeEvery=1, format='xyz', handle='writer')
#state.activateWriteConfig(writeconfig)

masses = 0
for a in state.atoms:
    masses += (a.mass * massO)
bounds = state.bounds.hi - state.bounds.lo
volume = bounds[0] * bounds[1] * bounds[2]
volume *= sig**3
volumeMeters = volume * (1e-30)
masses *= (1/6.022e23) * (1e-3)
density = masses / volumeMeters
print "density: " + str(density)

integVerlet = IntegratorVerlet(state)


# for computing density, change to real units
# mult by sig^3 - A^3 units

integVerlet.run(1000)

# calc final density
bounds = state.bounds.hi - state.bounds.lo
volume = bounds[0] * bounds[1] * bounds[2]
volume *= sig**3
volumeMeters = volume * (1e-30)
density = masses / volumeMeters
print "density: " + str(density)
