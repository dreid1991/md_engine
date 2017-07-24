import sys

sys.path = sys.path + ['/home/webbm/md_engine/core/build/python/build/lib.linux-x86_64-2.7' ]
sys.path.append('/home/webbm/md_engine/core/util_py')

from DASH import *
from LAMMPS_Reader import LAMMPS_Reader
import argparse
import re
import matplotlib.pyplot as plt
from math import *
state = State()
state.deviceManager.setDevice(0)
state.rCut = 10.0
state.padding = 2.0
state.periodicInterval = 7
state.shoutEvery = 100

anglePot = FixAngleHarmonic(state, 'angleHarm')

tempData = state.dataManager.recordTemperature('all','scalar', 100)

writeconfig = WriteConfig(state, fn='poly_out', writeEvery=100, format='xyz', handle='writer')
state.activateWriteConfig(writeconfig)
state.bounds = Bounds(state, lo = Vector(-10, -10, -10), hi = Vector(10, 10, 10))

integVerlet = IntegratorVerlet(state)

state.atomParams.addSpecies(handle='spc1',mass=1,atomicNum=1)
state.atomParams.setValues('spc1',mass=6,atomicNum=6)
state.addAtom('spc1',Vector(0.0,0.0,0.0))
state.addAtom('spc1',Vector(1.0,0.0,0.0))
state.addAtom('spc1',Vector(1.0,1.0,0.0))


bondQuart = FixBondQuartic(state, 'bondQuart')
bondQuart.setBondTypeCoefs(type=0, k2=607.19,k3=-1388.65,k4=1852.58, r0=0.9419)
bondQuart.createBond(state.atoms[0], state.atoms[1], type=0)
bondQuart.createBond(state.atoms[1], state.atoms[2], type=0)
state.activateFix(bondQuart)

anglePot.setAngleTypeCoefs(type=0, k=87.85, theta0=107.4/180.0*pi);
anglePot.createAngle(state.atoms[0], state.atoms[1], state.atoms[2], type=0)
state.activateFix(anglePot)
InitializeAtoms.initTemp(state,'all',1.2)


integVerlet.run(10000)







