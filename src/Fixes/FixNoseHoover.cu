#include "FixNoseHoover.h"
#include <cmath>
#include <string>

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include "cutils_func.h"
#include "cutils_math.h"
#include "Logging.h"
#include "State.h"
#include "Mod.h"

enum PRESSMODE {ISO, ANISO};
namespace py = boost::python;

std::string NoseHooverType = "NoseHoover";

// CUDA function to calculate the total kinetic energy

// CUDA function to rescale particle velocities
__global__ void rescale_cu(int nAtoms, uint groupTag, float4 *vs, float4 *fs, float3 scale)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint groupTagAtom = ((uint *) (fs+idx))[3];
        if (groupTag & groupTagAtom) {
            float4 vel = vs[idx];
            vel.x *= scale.x;
            vel.y *= scale.y;
            vel.z *= scale.z;
            vs[idx] = vel;
        }
    }
}

__global__ void rescale_no_tags_cu(int nAtoms, float4 *vs, float3 scale)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 vel = vs[idx];
        vel.x *= scale.x;
        vel.y *= scale.y;
        vel.z *= scale.z;
        vs[idx] = vel;
    }
}

// general constructor; may be a thermostat, or a barostat-thermostat
FixNoseHoover::FixNoseHoover(boost::shared_ptr<State> state_, std::string handle_,
                             std::string groupHandle_)
        : 
          Fix(state_,
              handle_,           // Fix handle
              groupHandle_,      // Group handle
              NoseHooverType,   // Fix name
              false,            // forceSingle
              false,
              false,            // requiresCharges
              1,                 // applyEvery
              50                // orderPreference
             ), 
                kineticEnergy(GPUArrayGlobal<float>(2)),
                ke_current(0.0), ndf(0),
                chainLength(3), nTimesteps(1), n_ys(1),
                pchainLength(3),
                nTimesteps_b(1), n_ys_b(1),
                weight(std::vector<double>(n_ys,1.0)),
                //thermPos(std::vector<double>(chainLength,0.0)),
                thermVel(std::vector<double>(chainLength,0.0)),
                thermForce(std::vector<double>(chainLength,0.0)),
                thermMass(std::vector<double>(chainLength,0.0)),
                scale(make_float3(1.0f, 1.0f, 1.0f)),
                omega(std::vector<double>(6)),
                omegaVel(std::vector<double>(6)),
                omegaMass(std::vector<double>(6)),
                pressFreq(6, 0),
                pressCurrent(6, 0),
                pFlags(6, false),
                tempComputer(state, "scalar"), 
                pressComputer(state, "scalar"), 
                pressMode(PRESSMODE::ISO),
                thermostatting(true),
                barostatting(false)
{
    pressComputer.usingExternalTemperature = true;

    // set flag 'requiresPostNVE_V' to true
    requiresPostNVE_V = true;

    // this is a thermostat (if we are barostatting, we are also thermostatting)
    isThermostat = true;
    identity = Virial(1,1,1,0,0,0); //as xx, yy, zz, xy, xz, yz
    // 
    barostatThermostatChainLengthSpecified = false;
    coupleStyleSpecified = false;
    barostattedDimensionsSpecified = false;

    // denote whether or not this is the first time prepareForRun was called
    // --- need this, because we need to initialize this with proper virials
    firstPrepareCalled = true;

}


void FixNoseHoover::parseKeyword(std::string keyword) {
    if (keyword == "ISO") {
        pressMode = PRESSMODE::ISO;
        couple = COUPLESTYLE::XYZ;
        // state of the pressComputer defaults to scalar (see the constructor above)
    } else if (keyword == "ANISO") {
        // allow the x, y, z dimensions to vary dynamically according to their instantaneous stress
        // --- so, still restricting to the hydrostatic pressure (1/3 * \sigma_{ii}) but
        //     not coupling.
        pressMode = PRESSMODE::ANISO;
        couple = COUPLESTYLE::NONE;
        // change the state of our pressComputer - and tempComputer - to "tensor"
        pressComputer = DataComputerPressure(state,"tensor");
        tempComputer = DataComputerTemperature(state,"tensor");
        // and assert again that we are using an external temperature computer (tempComputer) for
        // our pressure computer
        pressComputer.usingExternalTemperature = true;

    } else {
        barostatErrorMessage = "Invalid keyword in FixNoseHoover::setPressure():\n";
        barostatErrorMessage += "Valid options are \"ISO\", \"ANISO\";";
        printf(barostatErrorMessage.c_str());
        mdError("See above error message");

    };

    // regulating pressure for X, Y dims
    // set nDimsBarostatted to 2
    pFlags[0] = true;
    pFlags[1] = true;
    // set Z flag to true if we are not a 2d system
    if (! (state->is2d)) {
        pFlags[2] = true;
    }
}


// pressure can be constant double value
void FixNoseHoover::setPressure(std::string pressMode, double press, double timeConstant) {
    // get the pressmode and couplestyle; parseKeyword also changes the 
    // state of the pressComputer & tempComputer if needed; alters the boolean flags for 
    // the dimensions that will be barostatted (XYZ, or XY).
    parseKeyword(pressMode);
    pressInterpolator = Interpolator(press);
    barostatting = true;
    //pressFreq[0] = pressFreq[1] = pressFreq[2] = 1.0 / timeConstant;
    pFrequency = 1.0 / timeConstant;
}

// could also be a python function
void FixNoseHoover::setPressure(std::string pressMode, py::object pressFunc, double timeConstant) {
    parseKeyword(pressMode);
    pressInterpolator = Interpolator(pressFunc);
    barostatting = true;
    //pressFreq[0] = pressFreq[1] = pressFreq[2] = 1.0 / timeConstant;
    pFrequency = 1.0 / timeConstant;
}

// could also be a list of set points with accompanying intervals (denoted by turns - integer values)
void FixNoseHoover::setPressure(std::string pressMode, py::list pressures, py::list intervals, double timeConstant) {
    parseKeyword(pressMode);
    pressInterpolator = Interpolator(pressures, intervals);
    barostatting = true;
    //pressFreq[0] = pressFreq[1] = pressFreq[2] = 1.0 / timeConstant;
    pFrequency = 1.0 / timeConstant; 
}


// and analogous procedure with setting the temperature
void FixNoseHoover::setTemperature(double temperature, double timeConstant) {
    tempInterpolator = Interpolator(temperature);
    frequency = 1.0 / timeConstant;

}

void FixNoseHoover::setTemperature(py::object tempFunc, double timeConstant) {
    tempInterpolator = Interpolator(tempFunc);
    frequency = 1.0 / timeConstant;
}

void FixNoseHoover::setTemperature(py::list temps, py::list intervals, double timeConstant) {
    tempInterpolator = Interpolator(temps, intervals);
    frequency = 1.0 / timeConstant;
}



bool FixNoseHoover::prepareForRun()
{

    // if we are barostatting, we need the virials.
    // if this is the first time that prepareForRun was called, we do not have them
    // so, return false and it'll get called again
    if (firstPrepareCalled && barostatting) {
        firstPrepareCalled = false;
        return false;

    }

    // get our boltzmann constant
    boltz = state->units.boltz;

    // Calculate current kinetic energy
    tempInterpolator.turnBeginRun = state->runInit;
    tempInterpolator.turnFinishRun = state->runInit + state->runningFor;
    tempComputer.prepareForRun();
    
    calculateKineticEnergy();
    
    tempInterpolator.computeCurrentVal(state->runInit);
    updateMasses();

    // Update thermostat forces
    double temp = tempInterpolator.getCurrentVal();
    thermForce.at(0) = (ke_current - ndf * boltz * temp) / thermMass.at(0);
    for (size_t k = 1; k < chainLength; ++k) {
        thermForce.at(k) = (
                thermMass.at(k-1) *
                thermVel.at(k-1) *
                thermVel.at(k-1) - boltz*temp
            ) / thermMass.at(k);
    }

    // we now have the temperature set point value and instantaneous value.
    // -- set up the pressure set point value via pressInterpolator, 
    //    and get the instantaneous pressure.
    //   -- set initial values for assorted barostat and barostat-thermostat mass parameters
    //      analogous to what was done above
    if (barostatting) {
        // pressComputer, tempComputer were already set to appropriate "scalar"/"tensor" values
        // in the parseKeyword() call in the pertinent setPressure() function
        
        // get the number of dimensions barostatted
        nDimsBarostatted = 0;
        // go up to 6 - eventually we'll want Rahman-Parinello stress ensemble
        // -- for now, the max value of nDimsBarostatted is 3.
        for (int i = 0; i < 6; i++) {
            if (pFlags[i]) {
                nDimsBarostatted += 1;
            }
        }

        // set up our pressure interpolator
        pressInterpolator.turnBeginRun = state->runInit;
        pressInterpolator.turnFinishRun = state->runInit + state->runningFor;
        pressInterpolator.computeCurrentVal(state->runInit);

        // call prepareForRun on our pressComputer
        pressComputer.prepareForRun();

        // our P_{ext}, the set point pressure
        hydrostaticPressure = pressInterpolator.getCurrentVal();

        // using external temperature... so send tempNDF and temperature scalar/tensor 
        // to the pressComputer, then call [computeScalar/computeTensor]_GPU() 
        if (pressMode == PRESSMODE::ISO) {
            double scaledTemp = currentTempScalar;
            pressComputer.tempNDF = ndf;
            pressComputer.tempScalar = scaledTemp;
            pressComputer.computeScalar_GPU(true, groupTag);
        } else if (pressMode == PRESSMODE::ANISO) {

            Virial tempTensor_current = tempComputer.tempTensor;

            Virial scaledTemp = Virial(tempTensor_current[0], 
                                       tempTensor_current[1],
                                       tempTensor_current[2],
                                       0, 0, 0);

            pressComputer.tempNDF = ndf;
            pressComputer.tempTensor = scaledTemp;
            pressComputer.computeTensor_GPU(true, groupTag);

        } else {
            // to be implemented
            mdError("Only ISO and ANISO supported at this time");
        }

        // synchronize devices after computing the pressure..
        cudaDeviceSynchronize();

        // from GPU data, tell pressComputer to compute pressure on CPU side
        // --- might consider a boolean template argument for inside run() functions
        //     whether or not it is pressmode iso or aniso..
        //     --- once we go beyond iso&aniso we might make it class template
        if (pressMode == PRESSMODE::ISO) {
            pressComputer.computeScalar_CPU();
        } else {
            pressComputer.computeTensor_GPU();
        }

        // get the instantaneous pressure from pressComputer; store it locally as a tensor
        // --- some redundancy here if we are using a scalar
        getCurrentPressure();

        // following Julian's notation, denote barostat variables by press______ etc.


        // initialize the pressMass, pressVel, pressForce
        
        // TODO
        // and barostat thermostat variables: pressThermMass, pressThermVel, and pressThermForce, respectively





    }


    return true;
}
bool FixNoseHoover::postRun()
{
    tempInterpolator.finishRun();
    rescale();

    return true;
}

bool FixNoseHoover::stepInit()
{

    // see Martyna et. al. 2006 for clarification of notation, p. 5641
    // lists the complete propagator used here.

    // -- step init: update set points and associated variables before we do anything else
    if (barostatting) {

        // update the set points for:
        // - pressures    (--> and barostat mass variables accordingly)
        // - temperatures (--> barostat thermostat's masses, particle thermostat's masses)

        // save old values before computing new ones
        oldSetPointPressure = setPointPressure;
        oldSetPointTemperature = setPointTemperature;

        // compute set point pressure, and save it to our local variable setPointPressure
        pressInterpolator.computeCurrentVal(state->turn);
        setPointPressure = identity * pressInterpolator.getCurrentVal();

        // compute set point temperature, and save it to our local variable setPointTemperature
        tempInterpolator.computeCurrentVal(state->turn);
        setPointTemperature = identity * tempInterpolator.getCurrentVal();
        
        // compare values and update accordingly
        if (oldSetPointTemperature != setPointTemperature) {
            // update the masses associated with thermostats for the barostats and the particles
            updateBarostatMasses(true);
            updateBarostatThermalMasses(true);
            updateThermalMasses();
        }

        // exp(iL_{T_{BARO} \frac{\Delta t}{2})
        // -- variables that must be initialized/up-to-date:
        //    etaPressureMass, etaPressureVel, etaPressForce must all be /initialized (updated here)
        barostatThermostatIntegrate(true);

        // exp(iL_{T_{PART}} \frac{\Delta t}{2})
        // - does thermostat scaling of particle velocities
        thermostatIntegrate(true);

        // apply the scaling to the velocities
        rescale();

        // compute the kinetic energy of the particles
        // --- TODO: simple optimization here - we can save the scale factor and do the rescaling later
        calculateKineticEnergy();

        // a brief diversion from the propagators: we need to tell the GPU to do the summations of
        // of the virials to get the current pressure tensor
        if (pressMode == PRESSMODE::ISO) {
            pressComputer.tempNDF = ndf;
            pressComputer.tempScalar = currentTempScalar;
            pressComputer.computeScalar_GPU(true, groupTag);
            cudaDeviceSynchronize();
            pressComputer.computeScalar_CPU();
        } else if (pressMode == PRESSMODE::ANISO) {
            Virial tempTensor_current = tempComputer.tempTensor;
            pressComputer.tempNDF = ndf;
            pressComputer.tempTensor = tempTensor_current;
            pressComputer.computeTensor_GPU(true,groupTag);
            cudaDeviceSynchronize();
            pressComputer.computeScalar_CPU();
        }

        // and get the current pressure to our local variables
        getCurrentPressure();

        // and the current hydrostatic pressure
        hydrostaticPressure = pressInterpolator.getCurrentVal();

        // exp(iL_{\epsilon_2} \frac{\Delta t}{2})
        // -- barostat velocities from virial, including the $\alpha$ factor 1+ 1/N_f
        // -- note that we modified the kinetic energy above via the thermostat
        barostatVelocityIntegrate();

        // exp(iL_2 \frac{\Delta t}{2})
        // do barostat scaling of the particle momenta (velocities). 
        scaleVelocitiesBarostat();

        // after this, we exit stepInit, because we need IntegratorVerlet() to do a velocity timestep
        return true;

    } else {
        
        oldSetPointTemperature = setPointTemperature;
        tempInterpolator.computeCurrentVal();
        setPointTemperature = identity * tempInterpolator.getCurrentVal();
        // compare values and update accordingly
        
        if (oldSetPointTemperature != setPointTemperature) {
            // update the masses associated with thermostats for the particles
            updateThermalmasses();
        }

        thermostatIntegrate();
        

        return true;
        
    }

}

bool FixNoseHoover::postNVE_V() {
   
    if (barostatting) {
        rescaleVolume();
    }
    return true;

}

bool FixNoseHoover::postNVE_X() {
    if (barostatting) {
        rescaleVolume() 
    }

    return true;
}

bool FixNoseHoover::stepFinal()
{
    // at this point we have performed our second velocity verlet update of the particle velocities

    // - do barostat scaling of velocities
    if (barostatting) {
        
        // exp(iL_2 \frac{\Delta t}{2}) -- barostat rescaling of velocities component
        scaleVelocitiesBarostat();

        // exp(iL_{\epsilon_2} \frac{\Delta t}{2})
        // integration of barostat velocities
        barostatVelocityIntegrate();

        // exp(iL_{T_{PART}} \frac{\Delta t}{2})
        // scaling of particle velocities from particle thermostats
        thermostatIntegrate();

        // exp(iL_{T_{BARO}} \frac{\Delta t}{2})
        // scaling of barostat velocities from barostat thermostats
        barostatThermostatIntegrate(false);
    }

}

// save instantaneous pressure locally, and partition according to COUPLESTYLE::{XYZ,NONE}
void FixNoseHoover::getCurrentPressure() {

    // identity is a tensor made in constructor, since we use it throughout.
    // based off of the virial class
    if (pressMode == PRESSMODE::ISO) {
        double pressureScalar = pressComputer.pressureScalar;
        currentPressure = identity * pressureScalar;
    } else {
        Virial pressureTensor = pressComputer.pressureTensor;

        // partition the pressure;
        if (couple == COUPLESTYLE::XYZ) {
            // the virial pressure tensor in pressComputer goes as [xx,yy,zz,xy,xz,yz] 
            // (same as /src/Virial.h)
            double pressureScalar = (1.0 / 3.0) * (pressureTensor[0] + pressureTensor[1] + pressureTensor[2]);
            currentPressure = identity * pressureScalar;
        } else {
            currentPressure = pressureTensor;
            // explicitly set slant vectors to zero for now
            currentPressure[3] = currentPressure[4] = currentPressure[5] = 0.0;

        }
    }
}

// update barostat masses to reflect a change in the set point pressure
void FixNoseHoover::updateBarostatMasses(bool stepInit) {

    // set point temperature is of class Virial
    Virial t_external = setPointTemperature;
    if (stepInit) {
        // if we are at the initial step, use the old set point
        // -- this is due to ordering of the louiviliian propagators
        t_external = oldSetPointTemperature;
    }
    
    // the barostat mass expression is given in MTK 1994: Constant Pressure molecular dynamics algorithms
    // (1) isotropic: W = (N_f + d) kT / \omega_b^2
    // (2) anisotropic: W_g = W_g_0 = (N_f + d) kT / (d \omega_b^2)

    // 'N_f' number of degrees of freedom
    // -- held in our class variable ndf, from the tempComputer.ndf value
    //    (see ::calculateKineticEnergy())

    // 'd' - dimensionality of the system
    double d = 3.0;

    if (state->is2d) {
        d = 2.0;
    }

    Virial kt = boltz * t_external;
    // from MTK 1994
    if (pressMode == PRESSMODE::ISO) {
        // then we set the masses to case (1)
        pressMass = (ndf + d) * kt / (pFreq * pFreq);
    } else {
        pressMass = (ndf + d) * kt / (d * pFreq * pFreq);
    }
}

void FixNoseHoover::updateBarostatThermalMasses(bool stepInit) {

    // from MTK 1994:
    // Q_b_1 = d(d+1)kT/(2 \omega_b^2)
    double t_external = setPointTemperature[0];
    if (stepInit) {
        t_external = oldSetPointTemperature[0];
    }

    double kt = boltz * t_external;

    double d = 3.0;
    if (state->is2d) {
        d = 2.0;
    }

    pressThermMass[0] = d*(d+1.0) * kt / (2.0 * pFreq[0]);

    // Q_b_i = kt/(\omega_i^2)
    for (int i = 1; i < pchainLength; i++) {
        pressThermMass[i] = kt / (pFreq[i] * pFreq[i]);
    }

}

void FixNoseHoover::barostatThermostatIntegrate(bool stepInit) {

    // as thermostatIntegrate, get the set point temperature
    double kt = boltz * setPointTemperature;

    if (stepInit) {
        kt = boltz * oldSetPointTemperature;
    }

    // calculate the kinetic energy of our barostats - 
    //   only the dimensions we are barostatting.
    //   e.g., if 2D, we don't count Z (although it should be zero anyways)
    double ke_barostats = 0.0;
    for (int i = 0; i < 6; i++) {
        if (pFlags[i]) {
            ke_barostats += (pressMass[i] * pressVel[i] * pressVel[i]);
        }
    }

    // 
    
    if (!stepInit) {
        pressThermForce[0] = (ke_barostats - kt) / (pressThermMass[0]);
    }


    // this is the same routine as in thermostatIntegrate
    double ploop_weight = 1.0 / ( (double) nTimesteps_b);

    for (size_t i = 0; i < nTimesteps_b; ++i) {
        for (size_t j = 0; j < n_ys_b; ++j) {
            double timestep = weight.at(j)*state->dt / nTimesteps;
            double timestep2 = 0.5*timestep;
            double timestep4 = 0.25*timestep;
            double timestep8 = 0.125*timestep;

            // Update thermostat velocities
            pressThermVel.back() += timestep4*pressThermForce.back();
            for (size_t k = pchainLength-2; k > 0; --k) {
                double preFactor = std::exp( -timestep8*pressThermVel.at(k+1) );
                pressThermVel.at(k) *= preFactor;
                pressThermVel.at(k) += timestep4 * pressThermForce.at(k);
                pressThermVel.at(k) *= preFactor;
            }

            double preFactor = std::exp( -timestep8*pressThermVel.at(1) );
            pressThermVel.at(0) *= preFactor;
            pressThermVel.at(0) += timestep4*pressThermForce.at(0);
            pressThermVel.at(0) *= preFactor;

            // Update particle (barostat) velocities
            double barostatScaleFactor = std::exp( -timestep2*pressThermVel.at(0) );

            // apply the scaling of the barostat velocities
            for (int i = 0; i < 6; i++) {
                if (pFlags[i]) {
                    pressVel[i] *= barostatScaleFactor;
                }
            }

            // as done in particle thermostatting, get new ke_current (ke_barostats)
            ke_barostats = 0.0;
            for (int i = 0; i < 6; i++) {
                if (pFlags[i]) {
                    ke_barostats += (pressMass[i] * pressVel[i] * pressVel[i]);
                }
            }

            // Update the forces
            pressThermVel.at(0) *= preFactor;
            pressThermForce.at(0) = (ke_barostats - kt) / pressThermMass.at(0);
            pressThermVel.at(0) += timestep4 * pressThermForce.at(0);
            pressThermVel.at(0) *= preFactor;

            // Update thermostat velocities
            for (size_t k = 1; k < chainLength-1; ++k) {
                preFactor = std::exp( -timestep8*pressThermVel.at(k+1) );
                pressThermVel.at(k) *= preFactor;
                pressThermForce.at(k) = (
                        pressThermMass.at(k-1) *
                        pressThermVel.at(k-1) *
                        pressThermVel.at(k-1) - kt
                    ) / pressThermMass.at(k);
                pressThermVel.at(k) += timestep4 * pressThermForce.at(k);
                pressThermVel.at(k) *= preFactor;
            }

            pressThermForce.at(chainLength-1) = (
                    pressThermMass.at(chainLength-2) *
                    pressThermVel.at(chainLength-2) *
                    pressThermVel.at(chainLength-2) - kt
                ) / pressThermMass.at(chainLength-1);
            pressThermVel.at(chainLength-1) += timestep4*pressThermForce.at(chainLength-1);
        }
    }
}

void FixNoseHoover::thermostatIntegrate(bool stepInit) {
 // Equipartition at desired temperature
    // setPointTemperature should be up to date.
    double nkt = ndf * boltz * setPointTemperature[0];

    if (!stepInit) {
        thermForce.at(0) = (ke_current - nkt) / thermMass.at(0);
      //  printf("ke_current %f, nkt %f\n", ke_current, nkt);
    }

    //printf("temp %f boltz %f ndf %d\n", temp, boltz, int(ndf));
    // Multiple timestep procedure
    for (size_t i = 0; i < nTimesteps; ++i) {
        for (size_t j = 0; j < n_ys; ++j) {
            double timestep = weight.at(j)*state->dt / nTimesteps;
            double timestep2 = 0.5*timestep;
            double timestep4 = 0.25*timestep;
            double timestep8 = 0.125*timestep;

            // Update thermostat velocities
            thermVel.back() += timestep4*thermForce.back();
            for (size_t k = chainLength-2; k > 0; --k) {
                double preFactor = std::exp( -timestep8*thermVel.at(k+1) );
                thermVel.at(k) *= preFactor;
                thermVel.at(k) += timestep4 * thermForce.at(k);
                thermVel.at(k) *= preFactor;
            }

            double preFactor = std::exp( -timestep8*thermVel.at(1) );
            thermVel.at(0) *= preFactor;
            thermVel.at(0) += timestep4*thermForce.at(0);
            thermVel.at(0) *= preFactor;

            // Update particle velocities
            double scaleFactor = std::exp( -timestep2*thermVel.at(0) );
            scale *= scaleFactor;
            //printf("factor %f %f\n", scale.x, scaleFactor);

            ke_current *= scaleFactor*scaleFactor;

            // Update the thermostat positions
            //for (size_t k = 0; k < chainLength; ++k) {
            //    thermPos.at(k) += timestep2*thermVel.at(k);
            //}

            // Update the forces
            thermVel.at(0) *= preFactor;
            thermForce.at(0) = (ke_current - nkt) / thermMass.at(0);
            thermVel.at(0) += timestep4 * thermForce.at(0);
            thermVel.at(0) *= preFactor;

            // Update thermostat velocities
            for (size_t k = 1; k < chainLength-1; ++k) {
                preFactor = std::exp( -timestep8*thermVel.at(k+1) );
                thermVel.at(k) *= preFactor;
                thermForce.at(k) = (
                        thermMass.at(k-1) *
                        thermVel.at(k-1) *
                        thermVel.at(k-1) - boltz*temp
                    ) / thermMass.at(k);
                thermVel.at(k) += timestep4 * thermForce.at(k);
                thermVel.at(k) *= preFactor;
            }

            thermForce.at(chainLength-1) = (
                    thermMass.at(chainLength-2) *
                    thermVel.at(chainLength-2) *
                    thermVel.at(chainLength-2) - boltz*temp
                ) / thermMass.at(chainLength-1);
            thermVel.at(chainLength-1) += timestep4*thermForce.at(chainLength-1);
        }
    }

}

void FixNoseHoover::barostatVelocityIntegrate() {

    // $G_{\epsilon}$ = \alpha * (ke_current) + (Virial - P_{ext})*V

    // so, we need to have the /current pressure/
    //  we need to have the /current kinetic energy of the particles/
    //  we need to have the instantaneous volume
    //  also, note that the deformations of slant vectors are not affected 
    //  by the external pressure P_{ext}, should we incorporate this later

    // --- note: the anisotropy is incorporated
    // We evolve the barostat momenta according to this
    



}

void FixNoseHoover::scaleVelocitiesOmega() {
    float timestep4 = state->dt * 0.25;
    //so... if we're going to do triclinic this is going to get worse, b/c can't just lump all scale factors together into one number.  
    //see nh_v_press in LAMMPS
    float vals[3];
    for (int i=0; i<3; i++) {
        vals[i] = std::exp(-timestep4*(omegaVel[i] + mtkTerm2));
    }
    scale.x *= vals[0]*vals[0];
    scale.y *= vals[1]*vals[1];
    scale.z *= vals[2]*vals[2];

}

bool FixNoseHoover::halfStep(bool firstHalfStep)
{
    if (chainLength == 0) {
        mdWarning("Call of FixNoseHoover with zero thermostats in "
                  "the Nose-Hoover chain.");
        return false;
    }

    double boltz = state->units.boltz;

    // Update the desired temperature
    double temp;
    if (firstHalfStep) {
        double currentTemp = tempInterpolator.getCurrentVal();
        //printf("CURRENT TEMP IS %f\n", currentTemp);
        tempInterpolator.computeCurrentVal(state->turn);
        temp = tempInterpolator.getCurrentVal();
        //printf("SET PT IS %f\n", temp);
        if (currentTemp != temp) {
            updateMasses();
        }
    }

    // Get the total kinetic energy
    if (!firstHalfStep) {
        //! \todo This optimization assumes that the velocities are not changed
        //!       between stepFinal() and stepInit(). Can we add a check to make
        //!       sure this is indeed the case?
        temp = tempInterpolator.getCurrentVal();
        calculateKineticEnergy();
    }
    thermostatIntegrate(temp, boltz, firstHalfStep);

    if (barostatting) {
        //THIS WILL HAVE TO BE CHANGED FOR ANISO
        //if iso, all the pressure stuff is the same in every dimension
        /*
        if (pressMode == PRESSMODE::ISO) {
            double scaledTemp = tempScalar_current * scale.x;//keeping track of it for barostat so I don't have to re-sum
            pressComputer.tempNDF = ndf;
            pressComputer.tempScalar = scaledTemp;
            pressComputer.computeScalar_GPU(true, groupTag);
        } else if (pressMode == PRESSMODE::ANISO) {
            //not worrying about cross-terms for now
            Virial scaledTemp = Virial(tempTensor_current[0] * scale.x, tempTensor_current[1] * scale.y, tempTensor_current[2] * scale.z, 0, 0, 0);
            pressComputer.tempNDF = ndf;
            pressComputer.tempTensor = scaledTemp;
            pressComputer.computeScalar_GPU(true, groupTag);

        }
            */
        pressInterpolator.computeCurrentVal(state->turn);
        cudaDeviceSynchronize();
        if (pressMode == PRESSMODE::ISO) {
            pressComputer.computeScalar_CPU();
        } else if (pressMode == PRESSMODE::ANISO) {
            pressComputer.computeTensor_CPU();
        }
        setPressCurrent();
        omegaIntegrate();
        scaleVelocitiesOmega();
    }

    if (firstHalfStep) { 
        rescale();
    }
   
    // Update particle velocites
    //! \todo In this optimization, I assume that velocities are not accessed
    //!       between stepFinal() and stepInitial(). Can we ensure that this is
    //!       indeed the case?

    return true;
}


void FixNoseHoover::setPressCurrent() {
    for (int i=0; i<3; i++) {
        pressCurrent[i] = pressComputer.pressureScalar;
    }
    for (int i=3; i<6; i++) {
        pressCurrent[0] = pressComputer.pressureScalar;
    }

}
void FixNoseHoover::updateThermalMasses()
{
    double temp = tempInterpolator.getCurrentVal();
    thermMass.at(0) = ndf * boltz * temp / (frequency*frequency);
    for (size_t i = 1; i < chainLength; ++i) {
        thermMass.at(i) = boltz*temp / (frequency*frequency);
    }
}


void FixNoseHoover::calculateKineticEnergy()
{
    if (not barostatting) {
        tempComputer.computeScalar_GPU(true, groupTag);
        cudaDeviceSynchronize();
        tempComputer.computeScalar_CPU();
        ndf = tempComputer.ndf;
        ke_current = tempComputer.totalKEScalar;
    } else if (pressMode == PRESSMODE::ISO) {
        tempComputer.computeScalar_GPU(true, groupTag);
        cudaDeviceSynchronize();
        tempComputer.computeScalar_CPU();
        ndf = tempComputer.ndf;
        ke_current = tempComputer.totalKEScalar;

       // tempComputer.computeTensorFromScalar();
    } else if (pressMode == PRESSMODE::ANISO) {
        tempComputer.computeTensor_GPU(true, groupTag);
        cudaDeviceSynchronize();
        tempComputer.computeTensor_CPU();

        tempComputer.computeScalarFromTensor(); 
        ndf = tempComputer.ndf;
        ke_current = tempComputer.totalKEScalar;
        //need this for temp biz
    } 

    // set class variable currentTempScalar to value.
    // -- this way, it is always up-to-date (less scale factors, when those are implemented)
    currentTempScalar = tempComputer.tempScalar;
}

void FixNoseHoover::rescale()
{
    if (scale == make_float3(1.0f, 1.0f, 1.0f)) {
        return;
    }

    size_t nAtoms = state->atoms.size();
    if (groupTag == 1) {
        rescale_no_tags_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(
                                                 nAtoms,
                                                 state->gpd.vs.getDevData(),
                                                 scale);
    } else {
        rescale_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms,
                                                 groupTag,
                                                 state->gpd.vs.getDevData(),
                                                 state->gpd.fs.getDevData(),
                                                 scale);
    }

    scale = make_float3(1.0f, 1.0f, 1.0f);
}

Interpolator *FixNoseHoover::getInterpolator(std::string type) {
    if (type == "temp") {
        return &tempInterpolator;
    }
    return nullptr;
}
void export_FixNoseHoover()
{
    py::class_<FixNoseHoover,                    // Class
               boost::shared_ptr<FixNoseHoover>, // HeldType
               py::bases<Fix>,                   // Base class
               boost::noncopyable>
    (
        "FixNoseHoover",
        py::init<boost::shared_ptr<State>, std::string, std::string, py::object, double>(
            py::args("state", "handle", "groupHandle", "tempFunc", "timeConstant")
        )
    )
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, py::list, py::list, double>(
                py::args("state", "handle", "groupHandle", "intervals", "temps", "timeConstant")

                )
        )
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, double, double>(
                py::args("state", "handle", "groupHandle", "temp", "timeConstant")

                )
        )

    ;
}
