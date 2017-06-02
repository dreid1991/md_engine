Recording Data
==============

Overview
^^^^^^^^

Several types of data can be recorded using GPU-accelerated methods in DASH, including energies on an aggregate or fine-grained bases, temperature, kinetic energy, pressture, virial coefficients, volume, and simulation boundaries.  Other data types can be recorded via Python Operations (LINK) with minimal performance overhead.

Recording data
^^^^^^^^^^^^^^

The ``dataManager`` member of ``State`` handles data recording.  To record temperature of all atoms, for instance, one would write

.. code-block:: python

    #record temperature every 100 turns
    temperatureData = state.dataManager.recordTemperature(interval=100)
    
    integrater.run(10000)

    #print python list of recorded temperatures
    print temperature.vals
    #print python list of turns on which these values were recorded
    print temperature.turns

and as shown, access the recorded values through the ``vals`` member and the turns on which they were recorded through the ``turns`` member.  This structure is used for recording all data types.

Details on recording specific data types is given below.

Recording energies
^^^^^^^^^^^^^^^^^^

Recording temperatures and kinetic energies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recording pressures and virial coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recording volume and boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
