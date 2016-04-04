import sys
import matplotlib.pyplot as plt
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7', '../build/']
from Sim import *
state = State()
state.deviceManager.setDevice(0)
state.bounds = Bounds(state, lo = Vector(0, 0, 0), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000

state.grid = AtomGrid(state, 3.6, 3.6, 3.6)
state.atomParams.addSpecies(handle='spc1', mass=1, atomicNum=1)
nonbond = FixLJCut(state, 'cut', 'all')
nonbond.setParameter('sig', 'spc1', 'spc1', 1)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)
state.activateFix(nonbond)

f = open('init.xml').readlines()
for i in range(len(f)):
    bits = [float(x) for x in f[i].split()]
    state.addAtom('spc1', Vector(bits[0], bits[1], bits[2]))

InitializeAtoms.initTemp(state, 'all', 1.2)

fixNVT = FixNVTRescale(state, 'temp', 'all', [0, 1], [1.2, 1.2], 100)
state.activateFix(fixNVT)

integVerlet = IntegraterVerlet(state)

#tempData = state.dataManager.recordTemperature('all', 100)
#boundsData = state.dataManager.recordBounds(100)
#engData = state.dataManager.recordEnergy('all', 100)

#writeconfig = WriteConfig(state, fn='test_out', writeEvery=5000, format='xyz', handle='writer')
#state.activateWriteConfig(writeconfig)
integVerlet.run(10000)#000000)
sumV = 0.
for a in state.atoms:
    sumV += a.vel.lenSqr()
print sumV / len(state.atoms)/3.0

print boundsData.vals[0].getSide(1)
print engData.turns[-1]
print 'last eng %f' % engData.vals[-1]
print state.turn
print integVerlet.energyAverage('all')
perParticle = integVerlet.energyPerParticle()
print sum(perParticle) / len(perParticle)
