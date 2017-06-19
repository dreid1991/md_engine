Andersen Thermostat Fix
=======================

Overview
^^^^^^^^

Implements an Andersen thermostat for simulations to be held at a specified temperature as discussed in Andersen 1980 [Andersen1980]_.

Constructors
^^^^^^^^^^^^

.. code-block:: python

    FixNVTAndersen(state, handle, groupHandle, temp, nu, applyEvery)

Arguments 

``state``
    Simulation state to apply the fix.

``handle``
    A name for the object.  Named argument.

``groupHandle``
    Group of atoms to which the fix is applied.  Named argument.  Defaults to ``all``.

``temp``
    The temperature of the bath from which stochastic collisions are sampled.

``nu``
    The frequency of the sampling
    FixNVTAndersen(state, handle, groupHandle, tempFunc, nu, applyEvery)
    FixNVTAndersen(state, handle, groupHandle, intervals, temps, nu, applyEvery)


Examples
^^^^^^^^
