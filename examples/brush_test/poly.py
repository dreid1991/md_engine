import sys
sys.path = sys.path + ['/home/daniel/Documents/md_engine/core/build/python/build/lib.linux-x86_64-2.7' ]
sys.path.append('/home/daniel/Documents/md_engine/core/util_py')
import matplotlib.pyplot as plt
from LAMMPS_Reader import LAMMPS_Reader
from Sim import *
from math import *
state = State()
state.deviceManager.setDevice(0)
#state.bounds = Bounds(state, lo = Vector(-30, -30, -30), hi = Vector(360, 360, 360))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 10

state.dt = 0.005

ljcut = FixLJCut(state, 'ljcut')
bondFENE = FixBondFENE(state, 'bondFENE')
angleHarm = FixAngleHarmonic(state, 'angleHarm')
#tempData = state.dataManager.recordTemperature('all', 100)
state.activateFix(ljcut)
state.activateFix(bondFENE)
state.activateFix(angleHarm)

state.bounds = Bounds(state, Vector(-10, -10, -10), Vector(20, 24, 24))

writeconfig = WriteConfig(state, fn='poly_out', writeEvery=10, format='xyz', handle='writer')
#temp = state.dataManager.recordEnergy('all', collectEvery = 50)
#reader = LAMMPS_Reader(state=state, unitLen = unitLen, unitMass = 12, unitEng = 0.066, bondFix = bondHarm, angleFix = angleHarm, nonbondFix = ljcut, dihedralFix = dihedralOPLS, improperFix=improperHarm, atomTypePrefix = 'PTB7_', setBounds=False)
reader = LAMMPS_Reader(state=state, unitLen = 1, unitMass = 1, unitEng = 1, bondFix = bondFENE, nonbondFix = ljcut, angleFix = angleHarm, atomTypePrefix = 'POLY_', setBounds=False)
reader.read(dataFn = 'brush.data', inputFns = ['brush.in', 'brush.init', 'brush.settings'])
InitializeAtoms.initTemp(state, 'all', 0.1)

state.atomParams.setValues('POLY_0', atomicNum=6)
state.atomParams.setValues('POLY_1', atomicNum=7)
state.atomParams.setValues('POLY_2', atomicNum=1)
state.activateWriteConfig(writeconfig)

ewald = FixChargeEwald(state, "chargeFix", "all")
ewald.setParameters(64, 1.0, 3)
state.activateFix(ewald)


#integRelax = IntegratorRelax(state)
#integRelax.writeOutput()
#integRelax.run(11, 1e-9)
fixNVT = FixNoseHoover(state, 'temp', 'all', .1, 0.5)
state.activateFix(fixNVT)

integVerlet = IntegratorVerlet(state)
integVerlet.run(1500)







