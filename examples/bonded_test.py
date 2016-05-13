import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7', '../build/']
sys.path.append('../util_py')
from Sim import *
from math import *
state = State()
state.deviceManager.setDevice(1)
state.bounds = Bounds(state, lo = Vector(-10, -10, -10), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000
state.dt = .0005

state.grid = AtomGrid(state, 3.6, 3.6, 3.6)
state.atomParams.addSpecies(handle='spc1', mass=1, atomicNum=8)
nonbond = FixLJCut(state, 'cut')
nonbond.setParameter('sig', 'spc1', 'spc1', 1)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)
state.activateFix(nonbond)
#dihedral testing
state.addAtom('spc1', Vector(5, 5, 5))
state.addAtom('spc1', Vector(5, 6, 5))
state.addAtom('spc1', Vector(17, 6, 5))
state.addAtom('spc1', Vector(6, 5, 6))

bondHarm = FixBondHarmonic(state, 'bondHarm')
bondHarm.setBondTypeCoefs(type=0, k=15000, rEq=1.3);
bondHarm.setBondTypeCoefs(type=1, k=15000, rEq=1.3);
bondHarm.createBond(state.atoms[0], state.atoms[1], type=0)
bondHarm.createBond(state.atoms[1], state.atoms[2], type=1)
bondHarm.createBond(state.atoms[2], state.atoms[3], type=1)

state.activateFix(bondHarm)

dihedralOPLS = FixDihedralOPLS(state, 'dihedral')
dihedralOPLS.setDihedralTypeCoefs(type=0, coefs=[15, -10, 4, -12])

dihedralOPLS.createDihedral(state.atoms[0], state.atoms[1], state.atoms[2], state.atoms[3], coefs=[3,4,5,6])
state.activateFix(dihedralOPLS)
print dihedralOPLS.dihedrals
print dihedralOPLS.dihedrals[0].coefs[2]



angleHarm = FixAngleHarmonic(state, 'angHarm')
angleHarm.setAngleTypeCoefs(type=0, k=7000, thetaEq=2);
angleHarm.createAngle(state.atoms[0], state.atoms[1], state.atoms[2], thetaEq=3*pi/4, k=3)
angleHarm.createAngle(state.atoms[1], state.atoms[2], state.atoms[3], thetaEq=3*pi/4, k=3)
state.activateFix(angleHarm)
print angleHarm.angles[0].thetaEq
'''

#improper testing
state.addAtom('spc1', Vector(5, 5, 5))
state.addAtom('spc1', Vector(6, 6, 5))
state.addAtom('spc1', Vector(6, 7, 5))
state.addAtom('spc1', Vector(7, 5, 6))

bondHarm = FixBondHarmonic(state, 'bondHarm')
bondHarm.setBondTypeCoefs(type=0, k=10, rEq=1.3);
bondHarm.setBondTypeCoefs(type=1, k=10, rEq=1.3);

bondHarm.createBond(state.atoms[0], state.atoms[1], type=0)
bondHarm.createBond(state.atoms[1], state.atoms[2], type=1)
bondHarm.createBond(state.atoms[1], state.atoms[3], type=1)

state.activateFix(bondHarm)



angleHarm = FixAngleHarmonic(state, 'angHarm')
angleHarm.setAngleTypeCoefs(type=0, k=3, thetaEq=2);
angleHarm.createAngle(state.atoms[0], state.atoms[1], state.atoms[2], type=0)
angleHarm.createAngle(state.atoms[0], state.atoms[1], state.atoms[3], type=0)
angleHarm.createAngle(state.atoms[2], state.atoms[1], state.atoms[3], type=0)
state.activateFix(angleHarm)


improperHarmonic = FixImproperHarmonic(state, 'impHarm')
improperHarmonic.setImproperTypeCoefs(type=0, k=6, thetaEq=(106/180.)*pi)

improperHarmonic.createImproper(state.atoms[0], state.atoms[1], state.atoms[2], state.atoms[3], type=0)
state.activateFix(improperHarmonic)
'''
fixNVT = FixNVTRescale(state, 'temp', 'all', [0, 1], [.02, .02], 1000)
state.activateFix(fixNVT)

writeconfig = WriteConfig(state, fn='test_out', writeEvery=10, format='xyz', handle='writer')
state.activateWriteConfig(writeconfig)
integRelax = IntegraterRelax(state)
integRelax.run(1, 1e-9)
integVerlet = IntegraterVerlet(state)

#tempData = state.dataManager.recordTemperature('all', 100)
#boundsData = state.dataManager.recordBounds(100)
#engData = state.dataManager.recordEnergy('all', 100)

#integVerlet.run(100000)
