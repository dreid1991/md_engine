#include "FixNoseHoover.h"

#include <cmath>
#include <string>

#include <boost/python.hpp>

#include "cutils_func.h"
#include "cutils_math.h"
#include "Logging.h"
#include "State.h"
enum PRESSMODE {ISO, ANISO, TRICLINIC};
enum COUPLESTYLE {XYZ, XY, YZ, XZ}
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


// create a general instance of FixNoseHoover; this may be either
// a thermostat, or a barostat-thermostat
FixNoseHoover::FixNoseHoover(boost::shared_ptr<State> state_, std::string handle_,
                             std::string groupHandle_, double timeConstant_)
        : Fix(state_,
              handle_,           // Fix handle
              groupHandle_,      // Group handle
              NoseHooverType,   // Fix name
              false,            // forceSingle
              false,
              false,            // requiresCharges
              1                 // applyEvery
             ), frequency(1.0 / timeConstant_),
                kineticEnergy(GPUArrayGlobal<float>(2)),
                ke_current(0.0), ndf(0),
                chainLength(3), pchainLength(3),
                nTimesteps(1), n_ys(1),
                weight(std::vector<double>(n_ys,1.0)),
                //thermPos(std::vector<double>(chainLength,0.0)),
                thermVel(std::vector<double>(chainLength,0.0)),
                thermForce(std::vector<double>(chainLength,0.0)),
                thermMass(std::vector<double>(chainLength,0.0)),
                omega(std::vector<double>(6)),
                omegaVel(std::vector<double>(6)),
                omegaMass(std::vector<double>(6)),
                pressFreq(6, 0.0),
                pressCurrent(6, 0.0),
                pFlags(6, false),
                scale(make_float3(1.0f, 1.0f, 1.0f)),
                tempComputer(state, true, false), 
                pressComputer(state, true, false), 
                pressMode(PRESSMODE::ISO),
                thermostatting(true),
                barostatting(false)
{
    pressComputer.usingExternalTemperature = true;
    
    barostatThermostatChainLengthSpecified = false;



};

// we can set the pressure to be a double as the set point value;
void FixNoseHoover::setPressure(double press) {
    pressInterpolator = Interpolator(press);
    barostatting = true;
};

// the pressure may be a python function
void FixNoseHoover::setPressure(py::object pressFunc) {
    pressInterpolator = Interpolator(pressFunc);
    barostatting = true;

};

// the pressure may be a list of set points across assorted intervals
void FixNoseHoover::setPressure(py::list pressures, py::list intervals) {
    pressInterpolator = Interpolator(pressures,intervals); 
    barostatting = true;
};

// tempInterpolator can take a double as the set point value;
void FixNoseHoover::setTemperature(double temperature) {
    tempInterpolator = Interpolator(temperature);
};

// alternatively, it can take an actual python function
void FixNoseHoover::setTemperature(py::object tempFunc) {
    tempInterpolator = Interpolator(tempFunc);
};

// alternatively, it can take a list
void FixNoseHoover::setTemperature(py::list temps, py::list intervals) {
    tempInterpolator = Interpolator(temps,intervals);
};

void FixNoseHoover::setBarostatThermostatChainLength(int chainlength) {
    // the chainlength must be greater than or equal to 1; else, this fails
    assert(chainlength >= 1);
    etaPChainLength = chainlength;
    barostatThermostatChainLengthSpecified = true;
};

bool FixNoseHoover::verifyInputs() {
    /* THINGS WE MUST VERIFY:
     * 
     * - a barostatted dimension must be periodic
     * - (if triclinic) verify that the 2nd dimension of the off-diagonal component is periodic
     *              (e.g., if barostatting xz, verify z dimension is periodic)
     *      - for now, we must verify that the simulation is /not/ triclinic.
     * - if 2D simulation:
     *      - cannot barostat [z, xz,yz] dimensions
     *      - cannot specify coupling of XZ or YZ
     * - verify for coupling of dimensions that the coupled dimensions are barostatted
     *      for identical time intervals AND have the same frequencies
     * - all damping parameters - thermostat or barostat - must be strictly greater than zero
     *      (recommended value: ~100 timesteps)
     */
    
    // initialize the error message to something innocuous
    // -- note: if we failed anything, we'll have changed the error message and
    //          already returned false anyways; so this doesn't really matter.
    barostatErrorMessage = "Everything is fine here!";
    
    return true;
};

bool FixNoseHoover::prepareForRun()
{
    // Calculate current kinetic energy
    tempInterpolator.turnBeginRun = state->runInit;
    tempInterpolator.turnFinishRun = state->runInit + state->runningFor;
    tempComputer.prepareForRun();
    if (barostatting) {
        pressComputer.prepareForRun();
        pressComputer.usingExternalTemperature = true;
    }

    calculateKineticEnergy();
    tempInterpolator.computeCurrentVal(state->runInit);
    updateThermalMasses();

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

    // initialize the omega arrays
    if (barostatting) {

        if (!(verifyInputs() )) {
            //printf("%s\n", barostatErrorMessage);
            assert(false);
        };
        for (int i=0; i<6; i++) {
            omega[i] = 0;
            omegaVel[i] = 0;
            omegaMass[i] = temp * boltz * state->atoms.size() / (pressFreq[i] * pressFreq[i]); //
        }


        float3 dims = bounds->trace();
        // TODO: pressFreq needs to be populated with values! How? Idk.  What are sensible default values?
        //              -- what are the constraints, s.t. a dimension may be barostatted?
        //              -- what constitutes underspecified? overspecified?
        //              -- use assertions to verify that the system is properly specified?
        // TODO: get the maximum value of pressFreq; denote this as maxPressureFrequency
        

        // check if the barostat's thermostat chain length was specified;
        // if so, this flag will be true and etaPChainLength will have some finite value
        // else, set it to a default value of 3
        if (!(barostatThermostatChainLengthSpecified)) {
            etaPChainLength = 3;
        };
       
        double maxPressureFrequency = 0.0;
        // loop over pressFreq array and set maxPressureFrequency the maximum value;
        // it is safe to assume that all elements are greater than or equal to zero 
        // (if we are not barostatting a given dimension, it is left to the default value of zero)
        for (unsigned int jj = 0; jj < pressFreq.size(); jj++) {
            if (pressFreq[jj] > maxPressureFrequency) maxPressureFrequency = pressFreq[jj];
        };

        // initialize the etaPressure variables; and then populate with values
        // (these are the barostat's thermostat variables)
        etaPressure = std::vector<double> (etaPChainLength,0.0);
        etaPressure_dt = std::vector<double> (etaPChainLength, 0.0);
        etaPressure_dt2 = std::vector<double> (etaPChainLength, 0.0);
        etaPressure_mass = std::vector<double> (etaPChainLength, 0.0);
        
        // etaPressure_mass[j] = boltz * t_target / (p_freq_max^2);
        for (int j = 0; j < etaPChainLength; j++) {
            etaPressure[j] = boltz * temp / (maxPressureFrequency * maxPressureFrequency);
        };

        // get the initial volume of the simulation cell
        initialVolume = bounds->volume();
        
        // we want to use float3, not std::double<vector> (see return value of BoundsGPU *bounds ->()..
        // also, if we use GPU to compute anything, needs float3 not std::vector<double> which is really just a pointer
        refCell = bounds->trace();
        // // uncomment the following line when triclinic support is implemented
        //refCellSkews = bounds->skews();
        refCell_inv = bounds->invTrace();
        // // uncomment the following line when triclinic support is implemented; ensure Voigt notation is used in BoundsGPU
        //refCellSkews_inv = bounds->invSkews();
        // // sidenote: implement that ^^ function in BoundsGPU
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

// update the barostat positions (= volumes)
void FixNoseHoover::omegaIntegrate() {
    
    double boltz = state->units.boltz;
    double kt = boltz * temp;

    int nDims = 0;

    // get the number of dimensions which we barostat
    for (int i = 0; i < 6; i++) {
        nDims += (int) pFlags[i];
    };


    /*
    double timestep2 = 0.5 * state->dt;
    double forceOmega;
    double boltz = 1.0;
    double volume = state->boundsGPU.volume();
    int nDims = 0;
    for (int i=0; i<3; i++) {
        nDims += (int) pFlags[i];
    }
    if (pressMode == PRESSMODE::ISO) {
        mtkTerm1 = scale.x * scale.x * tempComputer * ndf * boltz / (state->atoms.size() * nDims); 
    } else {
        Virial mvSqr = tempComputer.tempTensor;
        for (int i=0; i<3; i++) {
            if (pFlags[i]) {
                mtkTerm1 += mvSqr[i];
            }
            mktTerm1 /= nDims * state->atoms.size();
        }
    }
    mtkTerm2 = 0;
    for (int i=0; i<3; i++) {
        if (pFlags[i]) {
            mtkTerm2 += omegaVel[i];
        }
    }
    mtkTerm2 /= nDims * state->atoms.size();
    //ignoring triclinic for now

    */


    
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
        tempInterpolator.computeCurrentVal(state->turn);
        temp = tempInterpolator.getCurrentVal();
        if (currentTemp != temp) {
            updateThermalMasses();
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
        // so, we need to update the masses; 
        // also, we need to update etap_mass according to t_target;
        // first
        if (pressMode == PRESSMODE::ISO) {
            double scaledTemp = tempScalar_current * scale.x;//keeping track of it for barostat so I don't have to re-sum
            pressComputer.tempNDF = ndf;
            pressComputer.tempScalar = scaledTemp;
            pressComputer.computeScalar_GPU(true, groupTag);
        } else {
            //not worrying about cross-terms for now
            Virial scaledTemp = Virial(tempTensor_current[0] * scale.x, tempTensor_current[1] * scale.y, tempTensor_current[2] * scale.z, 0, 0, 0);
            pressComputer.tempNDF = ndf;
            pressComputer.tempTensor = scaledTemp;
            pressComputer.computeScalar_GPU(true, groupTag);

        }
       
        pressInterpolator.computeCurrentVal(state->turn);
        cudaDeviceSynchronize();

        // if pressmode is iso, we only need a scalar value of the pressure
        if (pressMode == PRESSMODE::ISO) {
            pressComputer.computeScalar_CPU();
        } else {
            pressComputer.computeTensor_CPU();
        }
        setPressCurrent();
        
        omegaIntegrate();
        scaleVelocitiesOmega();
        */
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
    // store the computed pressure scalar (or tensor) locally within the pressCurrent vector

    if (pressMode == PRESSMODE::ISO) {
        for (int i = 0; i<3; i++) {
            pressCurrent[i] = pressComputer.pressureScalar;
        }
        return; // nothing more to do here...
    } else {
        Virial computedPressure = pressComputer.pressureTensor;
        if (couple == COUPLESTYLE::XYZ) {
            double val = (1.0 / 3.0) * (computedPressure[0] + computedPressure[1] + computedPressure[2]);
            for (int i = 0; i<3; i++) {
                pressCurrent[i] = val;
            }
        } else if (couple == COUPLESTYLE::XY) {
            double val = 0.5 * (computedPressure[0] + computedPressure[1]);
            pressCurrent[0] = pressCurrent[1] = val;
            pressCurrent[2] = computedPressure[2];
        } else if (couple == COUPLESTYLE::XZ) {
            double val = 0.5 * (computedPressure[0] + computedPressure[2]);
            pressCurrent[0] = pressCurrent[2] = val;
            pressCurrent[1] = computedPressure[1];
        } else if (couple = COUPLESTYLE::YZ) {
            double val = 0.5 * (computedPressure[1] + computedPressure[2]);
            pressCurrent[1] = pressCurrent[2] = val;
            pressCurrent[0] = computedPressure[0];
        } else {
            // no coupling
            pressCurrent[0] = computedPressure[0];
            pressCurrent[1] = computedPressure[1];
            pressCurrent[2] = computedPressure[2];
        }
    }

    /*
    if (pressMode == PRESSMODE::TRICLINIC) {
        // stuff here; not needed until we implement support for triclinic boxes
    };
    */
}
void FixNoseHoover::updateThermalMasses()
{
    // update the current thermal masses of the thermostat particles
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

        tempComputer.computeTensorFromScalar();
    } else if (pressMode == PRESSMODE::ANISO) {
        tempComputer.computeTensor_GPU(true, groupTag);
        cudaDeviceSynchronize();
        tempComputer.computeTensor_CPU();

        tempComputer.computeScalarFromTensor(); 
        ndf = tempComputer.ndf;
        ke_current = tempComputer.totalKEScalar;
        //need this for temp biz
    }
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

void FixNoseHoover::transformBox() 
{
// transform the simulation box by a prescribed volume s.t. NPT ensemble is sampled



};







void (FixNoseHoover::*setTemperature_x1)(double) = &FixNoseHoover::setTemperature;
void (FixNoseHoover::*setTemperature_x2)(py::object) = &FixNoseHoover::setTemperature;
void (FixNoseHoover::*setTemperature_x3)(py::list, py::list) = &FixNoseHoover::setTemperature;
void export_FixNoseHoover()   {

    py::class_<FixNoseHoover,                    // Class
               boost::shared_ptr<FixNoseHoover>, // HeldType
               py::bases<Fix>,                   // Base class
               boost::noncopyable>
    (   "FixNoseHoover",
        py::init<boost::shared_ptr<State>, std::string, std::string, double>(
            py::args("state", "handle", "groupHandle", "timeConstant")
            )
    )
    .def("setTemperature", setTemperature_x1, 
         (py::args("temperature")
       )
    )
    .def("setTemperature", setTemperature_x2,
         (py::arg("tempFunc")
         )
        )
    .def("setTemperature", setTemperature_x3,
         (py::arg("temps"), 
          py::arg("intervals")
         )
        )
    /*.def("setPressure", &FixNoseHoover::setPressure, 
         (py::arg("pressure")
         )
        )
        */
    // the old constructors; keep while we see if the set() methods work!
    /*
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
    */
    ;
}
