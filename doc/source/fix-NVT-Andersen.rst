Andersen Thermostat
===================================

Overview
^^^^^^^^


Constructor
^^^^^^^^^^^
.. code-block:: python
    FixNVTAndersen(state,handle,groupHandle,temp,nu,applyEvery)
    FixNVTAndersen(state,handle,groupHandle,tempFunc,nu,applyEvery)
    FixNVTAndersen(state,handle,groupHandle,intervals,temps,nu,applyEvery)



Arguments

``state``

``handle``

``groupHandle``

``temp``

``nu``

``applyEvery``

``tempFunc``

``intervals``

``temps``

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


