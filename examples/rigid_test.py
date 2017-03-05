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
state.shoutEvery = 100
state.dt = 0.0005

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

state.addAtom('spc2',Vector(4,4,1))
state.addAtom('spc1',Vector(5,5,1))
state.addAtom('spc2',Vector(6,4,1))

state.atoms[0].mass = 1.00794
state.atoms[1].mass = 15.99994
state.atoms[2].mass = 1.00794
state.atoms[0].vel = Vector(1,1,1)
state.atoms[1].vel = Vector(1,1,1)
state.atoms[2].vel = Vector(1,1,1)

rigid = FixRigid(state, 'rigid', 'all')

rigid.createRigid(1,0,2)

state.activateFix(rigid)

writeconfig = WriteConfig(state, fn='one_water', writeEvery=10, format='xyz', handle='writer')
state.activateWriteConfig(writeconfig)

integVerlet = IntegratorVerlet(state)
integVerlet.run(10000)
