from Sim import *
from random import random
from math import *

state = State()
state.deviceManager.setDevice(0)
state.periodicInterval = 10
state.rCut = 0.25 #need to implement padding
state.padding=0.02
T = 1.0
polymer_density=71.6*2
salt_density=71.6
polymer_length=140
bondlen=1.0/sqrt(polymer_length-1.0)
polymer_length=50
print "b", bondlen

boxsize=2.56


bnds=Bounds(state, lo=Vector(-0.5*boxsize, -0.5*boxsize, -0.5*boxsize), hi=Vector( 0.5*boxsize,  0.5*boxsize, 0.5*boxsize))
state.bounds = bnds
state.grid = AtomGrid(state, dx=0.32, dy=0.32, dz=0.32)#TODO 
state.atomParams.addSpecies(handle='A', mass=1)
state.atomParams.addSpecies(handle='B', mass=1)
state.atomParams.addSpecies(handle='S', mass=1)
state.atomParams.addSpecies(handle='P', mass=1)
state.atomParams.addSpecies(handle='N', mass=1)



ljcut = FixLJCut(state, handle='ljcut', groupHandle='all')

state.activateFix(ljcut)

ljcut.setParameter(param='eps', handleA='A', handleB='A', val=0.2)
ljcut.setParameter(param='sig', handleA='A', handleB='A', val=0.048)
ljcut.setParameter(param='eps', handleA='A', handleB='B', val=0.2)
ljcut.setParameter(param='sig', handleA='A', handleB='B', val=0.048)
ljcut.setParameter(param='eps', handleA='A', handleB='S', val=0.2)
ljcut.setParameter(param='sig', handleA='A', handleB='S', val=0.048)
ljcut.setParameter(param='eps', handleA='A', handleB='P', val=0.2)
ljcut.setParameter(param='sig', handleA='A', handleB='P', val=0.048)
ljcut.setParameter(param='eps', handleA='A', handleB='N', val=0.2)
ljcut.setParameter(param='sig', handleA='A', handleB='N', val=0.048)
ljcut.setParameter(param='eps', handleA='B', handleB='B', val=0.2)
ljcut.setParameter(param='sig', handleA='B', handleB='B', val=0.048)
ljcut.setParameter(param='eps', handleA='B', handleB='S', val=0.2)
ljcut.setParameter(param='sig', handleA='B', handleB='S', val=0.048)
ljcut.setParameter(param='eps', handleA='B', handleB='P', val=0.2)
ljcut.setParameter(param='sig', handleA='B', handleB='P', val=0.048)
ljcut.setParameter(param='eps', handleA='B', handleB='N', val=0.2)
ljcut.setParameter(param='sig', handleA='B', handleB='N', val=0.048)
ljcut.setParameter(param='eps', handleA='S', handleB='S', val=0.2)
ljcut.setParameter(param='sig', handleA='S', handleB='S', val=0.048)
ljcut.setParameter(param='eps', handleA='S', handleB='P', val=0.2)
ljcut.setParameter(param='sig', handleA='S', handleB='P', val=0.048)
ljcut.setParameter(param='eps', handleA='S', handleB='N', val=0.2)
ljcut.setParameter(param='sig', handleA='S', handleB='N', val=0.048)
ljcut.setParameter(param='eps', handleA='P', handleB='P', val=0.2)
ljcut.setParameter(param='sig', handleA='P', handleB='P', val=0.048)
ljcut.setParameter(param='eps', handleA='P', handleB='N', val=0.2)
ljcut.setParameter(param='sig', handleA='P', handleB='N', val=0.048)
ljcut.setParameter(param='eps', handleA='N', handleB='N', val=0.2)
ljcut.setParameter(param='sig', handleA='N', handleB='N', val=0.048)


#init from file
total_beads=25165
nchains=48

filename="initial_relaxed_L_"+str(boxsize)\
          +"_polymer_"+str(polymer_density)\
          +"_length_"+str(polymer_length)\
          +"_salt_"+str(salt_density)\
          +".dat"
ifile = open(filename, 'r') 

print "reading file ", filename
for i in range(total_beads):
    line= ifile.readline()
    words = line.split()
    state.addAtom(handle=words[3],pos=Vector(float(words[0]),float(words[1]),float(words[2])),q=float(words[4]))


bonds=FixBondHarmonic(state,handle="bond")
for i in range(nchains):
    for j in range(polymer_length-1):
        bonds.createBond(state.atoms[i*polymer_length+j], state.atoms[i*polymer_length+j+1], 3.0/pow(bondlen,2.0), 0.0)
ifile.close() 
state.activateFix(bonds)


#charge
charge=FixChargeEwald(state, "charge","all")
charge.setParameters(128,0.25,3);
state.activateFix(charge)




writer = WriteConfig(state, handle='writer', fn='coac_test_*', format='xyz', writeEvery=1000) 
state.activateWriteConfig(writer)
state.shoutEvery=1000
state.dt=0.0003

integrator = IntegratorLangevin(state,1.0)
integrator.set_params(1,1.0)
#integrator.run(2000000)
integrator.run(10000)

