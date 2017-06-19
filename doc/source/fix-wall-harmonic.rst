Harmonic Wall Potential
=======================

Overview
^^^^^^^^
Implements a harmonic potential energy wall at a specified origin ``origin`` with force in the vector direction ``forceDir``.  The cut off distance is specified by the ``dist`` keyword.  The spring constant associated with the wall is denoted by the ``k`` parameter.  The wall potential has a potential energy function given by 

.. math:: 
    U_{\text{wall}} = \frac{1}{2} k (r - r_{0})^2


Constructor
^^^^^^^^^^^
.. code-block:: python
    FixWallHarmonic(state,handle,groupHandle,origin,forceDir,dist,k)



Arguments

``state``
    Simulation state to apply the fix.

``handle``
    A name for the object.  String type.

``groupHandle``
    Group of atoms to which the fix is applied.  String type. 

``origin``
    Point of origin for the wall.  Vector type.   

``forceDir``
    The direction in which the force is to be applied.  Vector type.
``dist``
    The cutoff for the potential.  Float type.

``k``
    The spring constant associated with the wall.  Float type.

Python Member Functions
^^^^^^^^^^^^^^^^^^^^^^^

    Member data of the FixWallHarmonicpotential may be modified directly via the Python interface; namely, 'k', 'dist', 'forceDir', and 'origin' keywords are directly accessible from an instance of FixWallHarmonic.

    To modify any these parameters, simply assign them a new value of an appropriate type.

Examples
^^^^^^^^

