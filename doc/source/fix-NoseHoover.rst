Nose-Hoover Thermostat and Barostat
===================================

Overview
^^^^^^^^
Implements Nose-Hoover dynamics using the equations of motion as outlined in MTK 2006.  Allows for dynamics from the NVT or NPT ensembles to be simulated.  The exact propagator implemented for NPT dynamics is given by (math expression here)


Constructor
^^^^^^^^^^^
.. code-block:: python
    FixNoseHoover(state,handle,groupHandle)


Arguments

``state``
    The simulation state to which the fix is applied.

``handle``
    The name of the fix.  String type.

``groupHandle``
    The group of atoms to which this fix is to be applied.  String type.


Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^
The Nose-Hoover Barostat/Thermostat set points are set via the Python member functions.  

Specification of a set point temperature may be accomplished through any of the following commands:

.. code-block:: python
    setTemperature(temp,timeConstant)
    setTemperature(temps,intervals,timeConstant)
    setTemperature(tempFunc, timeConstant)

 Likewise, specification of a set point pressure may be accomplished through any of the following commands:

 .. code-block:: python
    setPressure(pressure,timeConstant)
    setPressure(pressures,intervals,timeConstant)
    setPressure(pressFunc,timeConstant)

 If NPT dynamics are desired, both ``setTemperature`` and ``setPressure`` should be called; the order in which they are called is immaterial.  

 
 Arguments: ``setTemperature``

 ``temp``
    The temperature set point for the simulation.  Double type.

 ``timeConstant``
    The time constant associated with the thermostat variables.  Double type.

 ``temps``
    A list of temperature set points to be used throughout the simulation.  List of doubles.
 
 ``intervals``
    A list of turns at which temperature set points change.  List of integers.
 
 ``tempFunc``
    The temperature set point, implemented as a python function. 

 Arguments: ``setPressure``

 ``pressure``
    The set point pressure for the simulation.  Double type.

 ``timeConstant``
    The time constant associated with the barostat variables.  Double type.

 ``pressures``
    A list of pressure set points to be used through the simulation.  List of doubles.
 
 ``intervals``
    A list of turns at which pressure set points change.  List of integers.
 
 ``pressFunc``
    The pressure set point, implemented as a python function.
 

Examples
^^^^^^^^

Example 1: Nose-Hoover Thermostat (NVT Ensemble) - constant set point temperature


Example 2: Nose-Hoover Barostat & Thermostat (NPT Ensemble) - constant set point temperature & pressure


Example 3: Nose-Hoover Barostat & Thermostat (NPT Ensemble) - constant set point temperature, python function pressure

