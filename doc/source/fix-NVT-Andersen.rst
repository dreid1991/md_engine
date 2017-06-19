Andersen Thermostat
===================================

Overview
^^^^^^^^
Implements the Andersen thermostat for maintaining a set point temperature through stochastic collisions with a heat bath at some set point temperature ``T``.

Constructors
^^^^^^^^^^^^
.. code-block:: python
    FixNVTAndersen(state,handle,groupHandle,temp,nu,applyEvery)
    FixNVTAndersen(state,handle,groupHandle,tempFunc,nu,applyEvery)
    FixNVTAndersen(state,handle,groupHandle,intervals,temps,nu,applyEvery)



Arguments

``state``
    The simulation State to which this fix is to be applied.

``handle``
    A name for this fix.  String type.

``groupHandle``
    The group of atoms to which the fix is applied.  String type.

``temp``
    The temperature of the heat bath (the set point temperature).  Double type.

``nu``
    A parameter describing the collision frequency of the system with the heat bath.  Float type.

``applyEvery``
    The number of turns to elapse between applications of this fix.  Integer type.

``tempFunc``
    The temperature of the heat bath, as a python function.  Python function.

``intervals``
    The turns corresponding to the intervals over which the list of temperature set points will be applied.  List of integers.

``temps``
    The list of temperature set points for the simulation.  List of floats.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^
The Andersen Thermostat allows for user-specification of the seed for the PRNG used to randomly sample from the heat bath.  If not specified, the seed takes a default value of 0.

Setting the seed for the PRNG is done with ``setParameters``:

.. code-block:: python
    setParameters(seed)

Arguments

``seed``
    The seed to be used by the PRNG.  Integer value.

Examples
^^^^^^^^


