import sys
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7']
from Sim import *
from math import *
from random import random


state = State()
#state.deviceManager.setDevice(0)
state.periodicInterval = 1
state.shoutEvery = 400000 #how often is says % done
state.rCut = 2.5 #need to implement padding
state.padding = 0.5
state.seedRNG()

# z bounds taken care of automatically in 2d simulation
state.bounds = Bounds(state, lo=Vector(0, 0, 0),
                             hi=Vector(60, 60, 60))
state.atomParams.addSpecies(handle='type1', mass=1)


initBoundsType1 = Bounds(state, lo=Vector(5,5,0),
                                hi=Vector(15,10,20))

InitializeAtoms.populateRand(state,bounds=initBoundsType1,
                            handle='type1', n=1, distMin = 1.0)


subTemp = 1.0
InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments
#writer = WriteConfig(state, handle='writer', fn='wallFix_test', format='xyz',
#                     writeEvery=2000)
#state.activateWriteConfig(writer)

state.dt = 0.1
alpha = 0.5
seed = 123985749
integrator = IntegratorLGJF(state,alpha,seed)
integrator.run(500000)

