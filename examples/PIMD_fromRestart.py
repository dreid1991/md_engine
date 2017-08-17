import sys
import os
import matplotlib.pyplot as plt

# relative path to /build/ dir
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
sys.path.append('../util_py')

from DASH import *
import water
from water import *
import math
import numpy as np

'''
This is an example script that shows how to restart a PIMD simulation of q-TIP4P/f water

In order to work, this script does require that you have run PIMD_TIP4P.py, found in the
current directory.

'''

# make an instance of State, and set the device we will be using
state = State()
state.deviceManager.setDevice(0)

# set periodicInterval, dt, rcut, and padding; this is required.
state.periodicInterval = 7
state.shoutEvery = 500
state.dt = 0.05
state.rCut = 12.0
state.padding = 1.0

# load the file, and iterate to the first configuration via the .next() method
state.readConfig.loadFile('tip4p_restart0.xml')
state.readConfig.next()

# instantiate and activate the assorted fixes...
# we were using LJ, Charge, FlexibleTIP4P, BondQuartic, and AngleHarmonic,
# along with a thermostat (but, thermostats require explicit creation)

charge = FixChargeEwald(state, 'charge', 'all')
charge.setParameters(128,state.rCut-1, 3)

nonbond = FixLJCut(state,'cut')
flexibleTIP4P = FixTIP4PFlexible(state,'TIP4PFlexible')
bondQuart = FixBondQuartic(state,'bondQuart')
harmonicAngle = FixAngleHarmonic(state,'angleH')

# activate the fixes
state.activateFix(charge)
state.activateFix(nonbond)
state.activateFix(flexibleTIP4P)
#state.activateFix(bondQuart)
state.activateFix(harmonicAngle)

# create a NoseHoover thermostat
fixNVT = FixNoseHoover(state,'nvt','all')
fixNVT.setTemperature(300.0, 200*state.dt)
state.activateFix(fixNVT)

# finally, make our integrator, and run for 5000 steps
integVerlet = IntegratorVerlet(state)

writer = WriteConfig(state, handle='writer', fn='restartPIMD', format='xyz',
                     writeEvery=10)
state.activateWriteConfig(writer)

writeRestart = WriteConfig(state, handle = 'restart',fn="tip4p_fromRestart*", format='xml',writeEvery=1000)
state.activateWriteConfig(writeRestart)

integVerlet.run(500)
