import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7', '../build/']
sys.path.append('../util_py')
import matplotlib.pyplot as plt
from LAMMPS_Reader import LAMMPS_Reader
from Sim import *
from math import *
state = State()
state.deviceManager.setDevice(1)
state.bounds = Bounds(state, lo = Vector(-10, -10, -10), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000
state.grid = AtomGrid(state, 3.6, 3.6, 3.6)


ljcut = FixLJCut(state, 'ljcut')
bondHarm = FixBondHarmonic(state, 'bondharm')
angleHarm = FixAngleHarmonic(state, 'angleHarm')

state.activateFix(ljcut)
#state.activateFix(bondHarm)
#state.activateFix(angleHarm)

unitLen = 3.55
writeconfig = WriteConfig(state, fn='dio_out', writeEvery=1, format='xyz', handle='writer')
writeconfig.unitLen = 1/unitLen
state.activateWriteConfig(writeconfig)

reader = LAMMPS_Reader(state=state, unitLen = unitLen, unitMass = 12, unitEng = 0.07, bondFix = bondHarm, angleFix = angleHarm, nonbondFix = ljcut, atomTypePrefix = 'DIO_', setBounds=False)
reader.read(dataFn = 'DIO_VMD.data')

state.atomParams.setValues('DIO_0', atomicNum=6)
state.atomParams.setValues('DIO_1', atomicNum=1)
state.atomParams.setValues('DIO_2', atomicNum=53)

integVerlet = IntegraterVerlet(state)

integVerlet.run(1)







