Nose-Hoover Thermostat and Barostat
===================================

Overview
^^^^^^^^
Implements Nose-Hoover dynamics using the equations of motion as outlined in MTK 2006.  Allows for dynamics from the NVT or NPT ensembles to be simulated.  The total Liouville operator implemented for NPT dynamics is given by (from MTK 2006, p.5641):

.. math:: 

    iL = iL_1 + iL_2 + iL_{\epsilon,1} + iL_{\epsilon,2} + iL_{T-baro} + iL_{T-Part}

    iL_1 = \sum\limits_{i=1}^N \bigl[\frac{\mathbf{p}_i}{m_i} + \frac{p_{\epsilon}}{W} \mathbf{r}_i \bigl] \cdot \frac{\partial}{\partial \mathbf{r}_i}

    iL_2 = \sum\limits_{i=1}^N \bigl[\mathbf{F}_i - \alpha \frac{p_{\epsilon}}{W}\mathbf{p}_i \bigl] \cdot \frac{\partial}{\partial \mathbf{p}_i}

    iL_{\epsilon,1} = \frac{p_{\epsilon}}{W} \frac{\partial}{\partial \epsilon}

    iL_{\epsilon,2} = G_{\epsilon} \frac{\partial}{\partial p_{\epsilon}}

    \text{where } G_{\epsilon} = \alpha \sum\limits_i \frac{\mathbf{p}_i^2}{m_i} + 
    \sum\limits_{i=1}^N \mathbf{r}_i \cdot \mathbf{F}_i - 3 V \frac{\partial U}{\partial V} - PV
   

Here, :math:`\mathbf{p}_i` and :math:`\mathbf{r}_i` are the particle momenta and positions, :math:`\mathbf{F}_i` are the forces on the particles, :math:`p_{\epsilon}` and :math:`W` are the barostat momenta and masses, :math:`\alpha` is a factor of :math:`1+\frac{1}{N}`, and :math:`P` and :math:`V` are the set point pressure and instantaneous volume, respectively.


The corresponding propagator for the NPT ensemble is then given by:

.. math:: 

    \exp(iL \Delta t) = \exp (iL_{T-baro} \frac{\Delta t}{2}) \exp (iL_{T-part} \frac{\Delta t}{2}) \exp (iL_{\epsilon,2} \frac{\Delta t}{2}) \\
    \times \exp (iL_2 \frac{\Delta t}{2}) \exp (iL_{\epsilon,1} \Delta t) \exp(iL_1 \Delta t) \exp(iL_2 \frac{\Delta t}{2}) \\
    \times \exp(iL_{\epsilon,2} \frac{\Delta t}{2}) \exp(iL_{T-part} \frac{\Delta t}{2}) \exp(iL_{T-baro} \frac{\Delta t}{2})

The barostat variables and particles are separately thermostatted; in each case, a chain of 3 thermostats is used.  Integration is accomplished via a first order Suzuki-Yoshida integration scheme.  

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

Arguments: 

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
 
    
    
Likewise, specification of a set point pressure may be accomplished through any of the following commands:

.. code-block:: python

   setPressure(mode,pressure,timeConstant)  
   setPressure(mode,pressures,intervals,timeConstant)
   setPressure(mode,pressFunc,timeConstant)

Arguments:

``mode``
    The mode in which cell deformations occur; options are "ISO" or "ANISO".  With mode "ISO", the internal stress tensor is averaged across the three normal components (or 2, for 2D simulations), and a uniform scale factor for the dimensions emerges.  For "ANISO", the components of the internal stress tensor are not averaged and the individual dimensions are scaled independently.

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


If NPT dynamics are desired, both ``setTemperature`` and ``setPressure`` should be called; the order in which they are called is immaterial.  



Examples
^^^^^^^^

Example 1: Nose-Hoover Thermostat (NVT Ensemble) - constant set point temperature

.. code-block:: python
    
    # set up a simulation state
    state = State()

    # make an instance of the fix
    fixNVT = FixNoseHoover(state,"nvt","all")

    # assign a set point temperature of 300K with time constant 10*state.dt
    fixNVT.setTemperature(300.0,10*state.dt)

    # activate the fix
    state.activateFix(fixNVT)


Example 2: Nose-Hoover Barostat & Thermostat (NPT Ensemble) - constant set point temperature & pressure

.. code-block:: python

    # set up a simulation state
    state = State()

    # make an instance of the fix
    fixNPT = FixNoseHoover(state,"npt","all")

    # assign a set point temperature and time constant 10*state.dt
    fixNPT.setTemperature(250.0,10*state.dt)

    # assign a set point pressure and time constant 1000*state.dt with isotropic cell deformations
    fixNPT.setPressure("ISO",1.0,1000*state.dt)

    # activate the fix
    state.activateFix(fixNPT)


.. rubric:: References


