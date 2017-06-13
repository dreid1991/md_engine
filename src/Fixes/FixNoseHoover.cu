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
    double boltz = state->units.boltz;
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

        // compute the hydrostatic pressure - average of $\sigma_{ii}$, i = 1,2,3
        computeHydrostaticPressure();

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
    return halfStep(true);
}

bool FixNoseHoover::stepFinal()
{
    return halfStep(false);
}

// save instantaneous pressure locally, and partition according to COUPLESTYLE::{XYZ,NONE}
void FixNoseHoover::getCurrentPressure() {

    // have some  identity tensor.. see /src/Virial.h: [xx,yy,zz,xy,xz,yz]
    Virial identity = Virial(1,1,1,0,0,0);
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



};
void FixNoseHoover::thermostatIntegrate(double temp, double boltz, bool firstHalfStep) {
 // Equipartition at desired temperature
    double nkt = ndf * boltz * temp;

    if (!firstHalfStep) {
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
void FixNoseHoover::updateMasses()
{
    double boltz = state->units.boltz;
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
