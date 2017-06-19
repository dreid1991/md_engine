Nose-Hoover Thermostat and Barostat
===================================

Overview
^^^^^^^^


Constructor
^^^^^^^^^^^
.. code-block:: python
    FixNoseHoover(state,handle,groupHandle)



Arguments

``state``

``handle``

``groupHandle``


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

 ``timeConstant``

 ``temps``

 ``intervals``

 ``tempFunc``

 Arguments: ``setPressure``

 ``pressure``

 ``timeConstant``

 ``pressures``

 ``intervals``

 ``pressFunc``

 

Examples
^^^^^^^^

Example 1: Nose-Hoover Thermostat (NVT Ensemble) - constant set point temperature


Example 2: Nose-Hoover Barostat & Thermostat (NPT Ensemble) - constant set point temperature & pressure


Example 3: Nose-Hoover Barostat & Thermostat (NPT Ensemble) - constant set point temperature, python function pressure

