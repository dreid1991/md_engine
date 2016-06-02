import sys
import matplotlib.pyplot as plt
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
#from Sim import *
from Sim import *
print FixPair
state = State()
state.deviceManager.setDevice(0)
state.bounds = Bounds(state, lo = Vector(0, 0, 0), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 100

state.grid = AtomGrid(state, 3.6, 3.6, 3.6)
state.atomParams.addSpecies(handle='spc1', mass=1, atomicNum=1)
nonbond = FixLJCut(state, 'cut')
nonbond.setParameter('sig', 'spc1', 'spc1', 1)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)
state.activateFix(nonbond)

f = open('init.xml').readlines()
for i in range(len(f)):
    bits = [float(x) for x in f[i].split()]
    state.addAtom('spc1', Vector(bits[0], bits[1], bits[2]))

#state.addAtom('spc1', pos = Vector(10, 10, 10))
#state.addAtom('spc1', pos = Vector(10.5, 10.5, 10.7))
#InitializeAtoms.initTemp(state, 'all', 1.2)

fixNVT = FixNoseHoover(state, 'temp', 'all', 1.2, 0.1)
#fixNVT = FixNVTRescale(state, 'temp', 'all', [0, 1], [1.2, .1], 1)
state.activateFix(fixNVT)

integVerlet = IntegratorVerlet(state)

#tempData = state.dataManager.recordTemperature('all', 10)
#boundsData = state.dataManager.recordBounds(100)
#engData = state.dataManager.recordEnergy('all', 100)

#writeconfig = WriteConfig(state, fn='test_out', writeEvery=5, format='xyz', handle='writer')
#state.activateWriteConfig(writeconfig)
integVerlet.run(10000)
#integVerlet.run(10000)
print state.atoms[0].pos
sumV = 0.
for a in state.atoms:
    sumV += a.vel.lenSqr()
print sumV / len(state.atoms)/3.0
#print tempData.vals
#print sum(tempData.vals) / len(tempData.vals)
#print boundsData.vals[0].getSide(1)
#print engData.turns[-1]
#print 'last eng %f' % engData.vals[-1]
#print state.turn
#print integVerlet.energyAverage('all')
#perParticle = integVerlet.energyPerParticle()
#print sum(perParticle) / len(perParticle)
