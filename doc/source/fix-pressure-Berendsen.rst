Berendsen Barostat
==================

Overview
^^^^^^^^
Implements the Berendsen barostat for maintaining pressure at a specified set point pressure ``P`` through rescaling of the volume every ``applyEvery`` turns. 

Constructor
^^^^^^^^^^^
.. code-block:: python
    FixPressureBerendsen(state,handle,pressure,period,applyEvery)

Arguments

``state``
    Simulation state to which this fix is applied.

``handle``
    A name for this fix.  String type.

``groupHandle``
    The group of atoms to which this fix is applied.  String type.

``pressure``
    The set point pressure for the simulation.  Double type.

``period``
    The time constant associated with the barostat.  Double type.

``applyEvery``
    The number of turns to elapse between applications of this fix.  Integer type.


Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    setParameters(maxDilation)

Arguments

``maxDilation``
    The maximum factor by which the system volume may be scaled in any given turn.  Defaults to 0.00001.

Examples
^^^^^^^^


