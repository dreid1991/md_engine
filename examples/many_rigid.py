import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7' ]
print sys.path
sys.path.append('../util_py')
from Sim import *
from math import *
import random
state = State()
state.deviceManager.setDevice(1)
state.bounds = Bounds(state, lo = Vector(-0, -0, -0), hi = Vector(20, 20, 20))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000000
state.dt = 0.0000005

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

rigid = FixRigid(state, 'rigid', 'all')

positions = []
for x in xrange(5):
    for y in xrange(5):
        for z in xrange(5):
            pos = Vector(x*4+1,y*4+1,z*4+1)
            positions.append(pos)

for i in xrange(100):
    position = positions[i]
    #orientation = Vector(random.uniform(0.0,1.0),random.uniform(0.0,1.0),random.uniform(0.0,1.0))
    #orientation /= orientation.len()
    atomO = position
    atomH1 = position + Vector(-0.757,-0.586,0)
    atomH2 = position + Vector(0.757,-0.586,0)
    state.addAtom('spc1', atomO)
    state.addAtom('spc2', atomH1)
    state.addAtom('spc2', atomH2)
    state.atoms[i*3].mass = 15.9994
    state.atoms[i*3+1].mass = 1.00794
    state.atoms[i*3+2].mass = 1.00794
    q = random.randint(0,2) - 1
    r = random.randint(0,2) - 1
    s = random.randint(0,2) - 1
    velocity = Vector(1.0*q,1.0*r,1.0*s)
    for j in range(3):
        state.atoms[i*3 + j].vel = velocity
    rigid.createRigid(i*3,i*3+2,i*3+1)

'''
state.addAtom('spc1',Vector(2.90064,2.29823,2.81643))
state.addAtom('spc2',Vector(2.14364,1.71223,2.81643))
state.addAtom('spc2',Vector(3.65764, 1.71223, 2.81643))

state.addAtom('spc1',Vector(6.64706, 5.96847, 2.55486))
state.addAtom('spc2',Vector(5.89006, 5.38247, 2.55486))
state.addAtom('spc2',Vector(7.40406, 5.38247, 2.55486))

state.addAtom('spc1',Vector(1.26577, 8.23594, 6.09964))
state.addAtom('spc2',Vector(0.508769, 7.64994, 6.09964))
state.addAtom('spc2',Vector(2.02277, 7.64994, 6.09964))

state.addAtom('spc1',Vector(6.39199, 5.69288, 7.90345))
state.addAtom('spc2',Vector(5.63499, 5.10688, 7.90345))
state.addAtom('spc2',Vector(7.14899, 5.10688, 7.90345))

state.addAtom('spc1',Vector(6.06717, 0.983654, 9.11693))
state.addAtom('spc2',Vector(5.31017, 0.397654, 9.11693))
state.addAtom('spc2',Vector(6.82417, 0.397654, 9.11693))

state.addAtom('spc1',Vector(1.51891, 7.16951, 7.80856))
state.addAtom('spc2',Vector(0.761915, 6.58351, 7.80856))
state.addAtom('spc2',Vector(2.27591, 6.58351, 7.80856))

state.addAtom('spc1',Vector(3.39484, 0.436597, 1.87147))
state.addAtom('spc2',Vector(2.63784, -0.149403, 1.87147))
state.addAtom('spc2',Vector(4.15184, -0.149403, 1.87147))

state.addAtom('spc1',Vector(6.37611, 5.3057, 4.44358))
state.addAtom('spc2',Vector(5.61911, 4.7197, 4.44358))
state.addAtom('spc2',Vector(7.13311, 4.7197, 4.44358))

state.addAtom('spc1',Vector(0.136785, 5.96677, 7.27561))
state.addAtom('spc2',Vector(-0.620215, 5.38077, 7.27561))
state.addAtom('spc2',Vector(0.893785, 5.38077, 7.27561))

state.addAtom('spc1',Vector(4.5816, 2.23202, 4.36386))
state.addAtom('spc2',Vector(3.8246, 1.64602, 4.36386))
state.addAtom('spc2',Vector(5.3386, 1.64602, 4.36386))

state.addAtom('spc1',Vector(1.90414, 3.00844, 8.60354))
state.addAtom('spc2',Vector(1.14714, 2.42244, 8.60354))
state.addAtom('spc2',Vector(2.66114, 2.42244, 8.60354))

state.addAtom('spc1',Vector(8.6359, 2.20825, 5.45768))
state.addAtom('spc2',Vector(7.8789, 1.62225, 5.45768))
state.addAtom('spc2',Vector(9.3929, 1.62225, 5.45768))

state.addAtom('spc1',Vector(5.57206, 9.55064, 4.99774))
state.addAtom('spc2',Vector(4.81506, 8.96464, 4.99774))
state.addAtom('spc2',Vector(6.32906, 8.96464, 4.99774))

state.addAtom('spc1',Vector(2.9481, 4.43456, 7.09747))
state.addAtom('spc2',Vector(2.1911, 3.84856, 7.09747))
state.addAtom('spc2',Vector(3.7051, 3.84856, 7.09747))

state.addAtom('spc1',Vector(9.46832, 5.08952, 2.25651))
state.addAtom('spc2',Vector(8.71132, 4.50352, 2.25651))
state.addAtom('spc2',Vector(10.2253, 4.50352, 2.25651))

vels = [Vector(0.208478, 0.990363, 0),Vector(0.107477, 0.760825, 0),Vector(0.256113, 0.780435, 0),Vector(0.495084, 0.524085, 0),Vector(1.70137, 0.825081, 0),Vector(0.133456, 0.310938, 0),Vector(0.799271, 0.985317, 0),Vector(1.68491, 1.62836, 0),Vector(1.41221, 0.0907879, 0),Vector(1.56291, 1.94734, 0),Vector(1.62347, 1.1369, 0),Vector(0.937016, 1.23907, 0),Vector(1.39545, 1.75719, 0),Vector(1.36431, 0.811151, 0),Vector(1.76838, 0.301742, 0)]

mol = 0
for mol in xrange(14):
    state.atoms[mol*3].mass = 15.9994
    state.atoms[mol*3+1].mass = 1.00794
    state.atoms[mol*3+2].mass = 1.00794
    state.atoms[mol*3].vel = vels[mol]
    state.atoms[mol*3+1].vel = vels[mol]
    state.atoms[mol*3+2].vel = vels[mol]

    rigid.createRigid(mol*3,mol*3+1,mol*3+2)
    mol += 1
'''
state.activateFix(rigid)

#saveconfig = WriteConfig(state, fn='many_record', writeEvery=10, format='xml', handle='saver')
#state.activateWriteConfig(saveconfig)

writeconfig = WriteConfig(state, fn='many_record', writeEvery=10000, format='xyz', handle='writer')
state.activateWriteConfig(writeconfig)

#integRelax.run(1, 1e-9)

count = 0
printout = ""                                                                                                                                                                                  
for a in state.atoms:
    printout += str(count) + " pos: " + str(a.pos) + " vel: " + str(a.vel) + "\n"
    count += 1

integVerlet = IntegratorVerlet(state)
integVerlet.run(100000000)

printout = ""
for a in state.atoms:
    printout += str(count) + " pos: " + str(a.pos) + " vel: " + str(a.vel) + "\n"
    count += 1

#print printout
