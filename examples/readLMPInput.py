import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7', '../build/']
sys.path.append('../util_py')
import matplotlib.pyplot as plt
from LAMMPS_Reader import LAMMPS_Reader
from Sim import *
from math import *
state = State()
state.deviceManager.setDevice(1)
state.bounds = Bounds(state, lo = Vector(-10, -10, -10), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000

bondHarm = FixBondHarmonic(state, 'bondharm')

reader = LAMMPS_Reader(state=state, unitLen = 3.55, unitMass = 12, unitEng = 0.07, bondFix = bondHarm)
reader.read(dataFn = 'DIO_VMD.data')




