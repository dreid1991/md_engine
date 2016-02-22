import sys
sys.path.append('../python')
from Sim import *

state = State()
state.bounds = Bounds(state, lo = Vector(0, 0, 0), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7

state.grid = AtomGrid(state, 3.6, 3.6, 3.6)
state.atomParams.addSpecies('spc1', 1)
nonbond = FixLJCut(state, 'cut', 'all')
nonbond.setParameter('sig', 'spc1', 'spc1', 1)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)
state.activateFix(nonbond)

f = open('init.xml').readlines()
for i in range(len(f)):
    bits = [float(x) for x in f[i].split()]
    state.addAtom('spc1', Vector(bits[0], bits[1], bits[2]))

InitializeAtoms.initTemp(state, 'all', 1.2)

fixNVT = FixNVTRescale(state, 'temp', 'all', [0, 1], [1.2, 1.2], 1000)
state.activateFix(fixNVT)

integVerlet = IntegraterVerlet(state)
integVerlet.run(30000)
sumV = 0.
for a in state.atoms:
    sumV += a.vel.lenSqr()
print sumV / len(state.atoms)/3.0
#integVerlet.run(30000)
