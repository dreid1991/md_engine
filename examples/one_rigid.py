import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7' ]
print sys.path
sys.path.append('../util_py')
from Sim import *
from math import *
import random
state = State()
state.deviceManager.setDevice(1)
state.bounds = Bounds(state, lo = Vector(-0, -0, -0), hi = Vector(10, 10, 10))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 10000
state.dt = 0.05

state.atomParams.addSpecies('spc1', 1, atomicNum=8)
state.atomParams.addSpecies('spc2', 1, atomicNum=1)

nonbond = FixLJCut(state, 'cut')
nonbond.setParameter('sig', 'spc1', 'spc1', 1)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)

nonbond.setParameter('sig', 'spc1', 'spc2', 1)
nonbond.setParameter('eps', 'spc1', 'spc2', 1)

nonbond.setParameter('sig', 'spc2', 'spc2', 1)
nonbond.setParameter('eps', 'spc2', 'spc2', 1)

state.activateFix(nonbond)

state.addAtom('spc2',Vector(3-0.757,3,1))
state.addAtom('spc1',Vector(3,3-0.586,1))
state.addAtom('spc2',Vector(3+0.757,3,1))

state.addAtom('spc2',Vector(2-0.757,3,3))
state.addAtom('spc1',Vector(2,3-0.586,3))
state.addAtom('spc2',Vector(2+0.757,3,3))

#state.addAtom('spc2',Vector(4.5-0.757,5,1))
#state.addAtom('spc1',Vector(4.5,5-0.586,1))
#state.addAtom('spc2',Vector(4.5+0.757,5,1))

state.atoms[1].mass = 15.9994
state.atoms[0].mass = 1.00794
state.atoms[2].mass = 1.00794
state.atoms[0].vel = Vector(0,0,0)
state.atoms[1].vel = Vector(0,0,0)
state.atoms[2].vel = Vector(0,0,0)
state.atoms[4].mass = 15.9994
state.atoms[3].mass = 1.00794
state.atoms[5].mass = 1.00794
state.atoms[3].vel = Vector(0,0,0)
state.atoms[4].vel = Vector(0,0,0)
state.atoms[5].vel = Vector(0,0,0)
#state.atoms[7].mass = 15.9994
#state.atoms[6].mass = 1.00794
#state.atoms[8].mass = 1.00794
#state.atoms[6].vel = Vector(0,0,0)
#state.atoms[7].vel = Vector(0,0,0)
#state.atoms[8].vel = Vector(0,0,0)

#state.addAtom('spc2',Vector(0,0,0))

#state.atoms[3].mass = 1.00794

rigid = FixRigid(state, 'rigid', 'all')

rigid.createRigid(1,0,2)
rigid.createRigid(4,3,5)
#rigid.createRigid(7,6,8)

state.activateFix(rigid)

writeconfig = WriteConfig(state, fn='one_water', writeEvery=1, format='xyz', handle='writer')
state.activateWriteConfig(writeconfig)

integVerlet = IntegratorVerlet(state)
integVerlet.run(10000)
#integVerlet.run(2)

m = 0
v = Vector(0,0,0)
for i in state.atoms:
    m += i.mass
    v += i.vel
kinetic_energy  = 0.5*m*v.lenSqr()
print kinetic_energy
