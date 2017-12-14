import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']

from DASH import *
state = State()
state.deviceManager.setDevice(0)
state.bounds = Bounds(state, lo = Vector(0, 0, 0), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 100

state.atomParams.addSpecies(handle='spc1', mass=1, atomicNum=1)
nonbond = FixLJCutFS(state, 'cut')
nonbond.setParameter('sig', 'spc1', 'spc1', 1)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)
state.activateFix(nonbond)

f = open('init.xml').readlines()
for i in range(len(f)):
    bits = [float(x) for x in f[i].split()]
    state.addAtom('spc1', Vector(bits[0], bits[1], bits[2]))

#state.addAtom('spc1', pos = Vector(10, 10, 10))
#state.addAtom('spc1', pos = Vector(10.5, 10.5, 10.7))
InitializeAtoms.initTemp(state, 'all', 1.2)

#fixNVT = FixNVTRescale(state, 'temp', 'all', 1.2)
fixNPT = FixNoseHoover(state,'npt','all')
fixNPT.setTemperature(1.2,100.0*state.dt)
#fixNPT.setPressure('ANISO',0.2,1000*state.dt)
state.activateFix(fixNPT)


integVerlet = IntegratorVerlet(state)

#writeconfig = WriteConfig(state, fn='test_out', writeEvery=20, format='xyz', handle='writer')
#state.activateWriteConfig(writeconfig)

state.tuning = False

# equilibrate for 50k turns
integVerlet.run(50000)

# deactivate the thermostat
state.deactivateFix(fixNPT)

# run NVE for 50 000 000 turns
integVerlet.run(1000000)


# conservedData will have a scalar of the conserved quantity (in NVE, KE + PE) in the system
# --- note that we do not pass in 'all' - this data recorder only records for the entire system no matter what
conservedData = state.dataManager.recordHamiltonian('scalar',100)
# tempdata will have a vector of the instantaneous kinetic energy in the system
tempData = state.dataManager.recordTemperature('all','vector', 100)
# engData will have a scalar of the total PE in the system
engData = state.dataManager.recordEnergy('all', 100)

energyFile = open('Benchmark_Test_Results.dat','w')

# ok, so we want the temperature ( = sum(tempData[index].vals) / len(tempData[index].vals))
# the total kinetic energy in the system ( = ????
# the total potential energy in the system ( = engData.vals[index] )
# the conserved quantity (= conservedData.vals[index]
boltz = 1.0
energyFile.write('{:>18s}  {:>18s}  {:>18s}  {:>18s}  {:>18s}\n'.format("Temperature (K)","Potential Energy","Kinetic Energy","Hamiltonian", "DataComputerH"))
    pe = enerData.vals[index]
    # if tempData records a vector, it is per-particle temperatures.
    # so, sum these up, then divide by N to get average temperature.
    temp = sum(tempData.vals[index]) / (float (len(tempData.vals[index])))
    # next, sum them up, and multiply by 1.5 - in the list is mvv / 3kb;
    # for LJ units, kb is 1; and we want sum(1/2 mvv)
    ke  = (3.0 / 2.0) * sum(tempData.vals[index])
    # the hamiltonian is just pe + ke
    hamiltonian = pe + ke
    # and the conserved quantity should be identically the value in hamiltonian
    conserved = conservedData.vals[index]
    energyFile.write('{:<18.14f}  {:>18.14f}  {:>18.14f}  {:>18.14f}  {:>18.14f}\n'.format(temp,pe,ke,hamiltonian,conserved))



