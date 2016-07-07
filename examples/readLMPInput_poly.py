import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7' ]
sys.path.append('../util_py')
import matplotlib.pyplot as plt
from LAMMPS_Reader import LAMMPS_Reader
from Sim import *
from math import *
state = State()
state.deviceManager.setDevice(0)
state.bounds = Bounds(state, lo = Vector(-30, -30, -30), hi = Vector(360, 360, 360))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000

state.dt = 0.0005

ljcut = FixLJCut(state, 'ljcut')
bondHarm = FixBondHarmonic(state, 'bondharm')
angleHarm = FixAngleHarmonic(state, 'angleHarm')
dihedralOPLS = FixDihedralOPLS(state, 'opls')
improperHarm = FixImproperHarmonic(state, 'imp')

tempData = state.dataManager.recordTemperature('all', 100)
state.activateFix(ljcut)
state.activateFix(bondHarm)
state.activateFix(angleHarm)
state.activateFix(dihedralOPLS)
state.activateFix(improperHarm)

unitLen = 3.5
writeconfig = WriteConfig(state, fn='poly_out', writeEvery=100, format='xyz', handle='writer')
writeconfig.unitLen = 1/unitLen
temp = state.dataManager.recordEnergy('all', collectEvery = 50)
#reader = LAMMPS_Reader(state=state, unitLen = unitLen, unitMass = 12, unitEng = 0.066, bondFix = bondHarm, angleFix = angleHarm, nonbondFix = ljcut, dihedralFix = dihedralOPLS, improperFix=improperHarm, atomTypePrefix = 'PTB7_', setBounds=False)
reader = LAMMPS_Reader(state=state, unitLen = unitLen, unitMass = 12, unitEng = 0.066, bondFix = bondHarm, nonbondFix = ljcut,  angleFix = angleHarm, dihedralFix = dihedralOPLS,improperFix=improperHarm,atomTypePrefix = 'PTB7_', setBounds=False)
reader.read(dataFn = 'poly_min.data')

InitializeAtoms.initTemp(state, 'all', 0.1)

'''
1 12
2 32.065
3 12
4 19
5 1
6 12
7 16
8 16
9 12
10 1
11 35.453
12 12
13 1
14 126.904
'''
state.atomParams.setValues('PTB7_0', atomicNum=6)
state.atomParams.setValues('PTB7_1', atomicNum=16)
state.atomParams.setValues('PTB7_2', atomicNum=6)
state.atomParams.setValues('PTB7_3', atomicNum=9)
state.atomParams.setValues('PTB7_4', atomicNum=1)
state.atomParams.setValues('PTB7_5', atomicNum=6)
state.atomParams.setValues('PTB7_6', atomicNum=8)
state.atomParams.setValues('PTB7_7', atomicNum=8)
state.atomParams.setValues('PTB7_8', atomicNum=6)
state.atomParams.setValues('PTB7_9', atomicNum=1)
state.atomParams.setValues('PTB7_10', atomicNum=17)
state.atomParams.setValues('PTB7_11', atomicNum=6)
state.atomParams.setValues('PTB7_12', atomicNum=1)
state.atomParams.setValues('PTB7_13', atomicNum=53)

integRelax = IntegratorRelax(state)
integRelax.writeOutput()
#integRelax.run(11, 1e-9)
InitializeAtoms.initTemp(state, 'all', 0.1)
fixNVT = FixNVTRescale(state, 'temp', 'all', [0, 1], [0.1, 3.8], 100)
state.activateFix(fixNVT)

integVerlet = IntegratorVerlet(state)
integVerlet.run(1500)

state.activateWriteConfig(writeconfig)
state.createMolecule([a.id for a in state.atoms])
print len(state.atoms)
for i in range(5):
    state.duplicateMolecule(state.molecules[-1])
    print state.molecules
    state.molecules[-1].translate(Vector(0, 0, 8))
integVerlet.run(150000)
print [x / len(state.atoms) for x in temp.vals]

#integVerlet = IntegraterVerlet(state)
#integVerlet.run(100000)
#print state.atoms[0].pos.dist(state.atoms[1].pos)
#print tempData.vals







