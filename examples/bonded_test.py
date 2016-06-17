import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7', '../build/']
print sys.path
sys.path.append('../util_py')
from Sim import *
from math import *
"""
state = State()
state.deviceManager.setDevice(1)
state.bounds = Bounds(state, lo = Vector(-10, -10, -10), hi = Vector(60, 60, 60))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000
state.dt = .005
state.grid = AtomGrid(state, 3.6, 3.6, 3.6)
state.atomParams.addSpecies(handle='spc1', mass=1, atomicNum=8)

nonbond = FixLJCut(state, 'ljcut')
nonbond.setParameter('sig', 'spc1', 'spc1', 2)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)
state.activateFix(nonbond)

#dihedral testing
state.addAtom('spc1', Vector(5, 5, 5))
state.addAtom('spc1', Vector(5, 6, 5))
state.addAtom('spc1', Vector(6, 6, 5))
state.addAtom('spc1', Vector(6, 5, 6))

#bondHarm = FixBondHarmonic(state, 'bondHarm')
#bondHarm.setBondTypeCoefs(type=0, k=15, rEq=1.3);
#bondHarm.setBondTypeCoefs(type=1, k=15, rEq=1.3);
#bondHarm.createBond(state.atoms[0], state.atoms[1], type=0)
#bondHarm.createBond(state.atoms[1], state.atoms[2], type=1)
#bondHarm.createBond(state.atoms[2], state.atoms[3], type=1)

#state.activateFix(bondHarm)

#dihedralOPLS = FixDihedralOPLS(state, 'dihedral')
#dihedralOPLS.setDihedralTypeCoefs(type=0, coefs=[15, -10, 4, -12])
dihedralOPLS = FixDihedralOPLS(state, 'dihedral')
dihedralOPLS.setDihedralTypeCoefs(type=0, coefs=[15, -10, 4, -12])


dihedralOPLS.createDihedral(state.atoms[0], state.atoms[1], state.atoms[2], state.atoms[3], type=0)
state.activateFix(dihedralOPLS)
#print dihedralOPLS.dihedrals
#print dihedralOPLS.dihedrals[0].coefs[2]

angleHarm = FixAngleHarmonic(state, 'angHarm')
angleHarm.setAngleTypeCoefs(type=0, k=7000, thetaEq=2);
angleHarm.createAngle(state.atoms[0], state.atoms[1], state.atoms[2], type=0)#thetaEq=3*pi/4, k=3)
#angleHarm.createAngle(state.atoms[1], state.atoms[2], state.atoms[3], thetaEq=3*pi/4, k=3)
angleHarm.createAngle(state.atoms[0], state.atoms[1], state.atoms[2], k=600, thetaEq=234134)
angleHarm.createAngle(state.atoms[0], state.atoms[1], state.atoms[3], type=0)
angleHarm.createAngle(state.atoms[2], state.atoms[1], state.atoms[3], type=0)
state.activateFix(angleHarm)
print angleHarm.angles[0].thetaEq



#improper testing
state.addAtom('spc1', Vector(5, 5, 5))
state.addAtom('spc1', Vector(6, 6, 5))
state.addAtom('spc1', Vector(6, 7, 5))
state.addAtom('spc1', Vector(7, 5, 6))

bond = FixBondHarmonic(state, 'bond')
bond.setBondTypeCoefs(type=0, k=10, rEq=1.3);
bond.setBondTypeCoefs(type=1, k=10, rEq=1.3);
bond.createBond(state.atoms[0], state.atoms[1], type=0)
bond.createBond(state.atoms[1], state.atoms[2], type=1)
bond.createBond(state.atoms[1], state.atoms[3], type=1)

state.activateFix(bond)



angleHarm = FixAngleHarmonic(state, 'angHarm')
angleHarm.setAngleTypeCoefs(type=0, k=3, thetaEq=2);
angleHarm.createAngle(state.atoms[0], state.atoms[1], state.atoms[2], k=600, thetaEq=234134)
angleHarm.createAngle(state.atoms[0], state.atoms[1], state.atoms[3], type=0)
angleHarm.createAngle(state.atoms[2], state.atoms[1], state.atoms[3], type=0)
state.activateFix(angleHarm)


improperHarmonic = FixImproperHarmonic(state, 'impHarm')
improperHarmonic.setImproperTypeCoefs(type=0, k=6, thetaEq=(106/180.)*pi)

improperHarmonic.createImproper(state.atoms[0], state.atoms[1], state.atoms[2], state.atoms[3], type=0)
state.activateFix(improperHarmonic)

fixNVT = FixNVTRescale(state, 'temp', 'all', [0, 1], [.02, .02], 1000)
state.activateFix(fixNVT)

writeconfig = WriteConfig(state, fn='test_write', writeEvery=10, format='xml', handle='writer')
state.activateWriteConfig(writeconfig)

#integRelax.run(1, 1e-9)
integVerlet = IntegratorVerlet(state)

#tempData = state.dataManager.recordTemperature('all', 100)
#boundsData = state.dataManager.recordBounds(100)
#engData = state.dataManager.recordEnergy('all', 100)

integVerlet.run(1)
"""
state = State()
state.deviceManager.setDevice(1)
state.bounds = Bounds(state, lo = Vector(-10, -10, -10), hi = Vector(60, 60, 60))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000
state.dt = .005
state.grid = AtomGrid(state, 3.6, 3.6, 3.6)
state.atomParams.addSpecies(handle='spc1', mass=1, atomicNum=8)

state.addAtom('spc1', Vector(5, 5, 5))
state.addAtom('spc1', Vector(5, 6, 5))
state.addAtom('spc1', Vector(6, 6, 5))
state.addAtom('spc1', Vector(6, 5, 6))

readconfig = state.readConfig
readconfig.loadFile("test_out.xml")
readconfig.next()

nonbond = FixLJCut(state, "ljcut")
state.activateFix(nonbond)
print "     FIXLJCUT Information"
print nonbond.restartChunk()
print

dihedralOPLS = FixDihedralOPLS(state, 'dihedral')
state.activateFix(dihedralOPLS)
print "     DIHEDRALOPLS Information"
print "----------------------------------"
print str(len(dihedralOPLS.dihedrals)) + " dihedrals"
print

angleHarmonic = FixAngleHarmonic(state, 'angHarm')
state.activateFix(angleHarmonic)

print "    ANGLEHARMONIC Information"
print "----------------------------------"
print str(len(angleHarmonic.angles)) + " angles"
print

improperHarmonic = FixImproperHarmonic(state, 'impHarm')
state.activateFix(improperHarmonic)

print "    IMPROPERHARMONIC Information"
print "----------------------------------"
print str(len(improperHarmonic.impropers)) + " impropers"
print

writeconfig = WriteConfig(state, fn='test_read', writeEvery=10, format='xml', handle='writer')
state.activateWriteConfig(writeconfig)
integVerlet = IntegratorVerlet(state)
integVerlet.run(2)

