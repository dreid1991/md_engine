#include "FixNoseHoover.h"

#include <cmath>
#include <string>

#include <boost/python.hpp>

#include "cutils_func.h"
#include "Logging.h"
#include "State.h"
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

FixNoseHoover::FixNoseHoover(boost::shared_ptr<State> state_, std::string handle_,
                             std::string groupHandle_, double temp_, double timeConstant_)
        : tempInterpolator(temp_),
          Fix(state_,
              handle_,           // Fix handle
              groupHandle_,      // Group handle
              NoseHooverType,   // Fix name
              false,            // forceSingle
              false,            // requiresVirials 
              false,            // requiresCharges
              1                 // applyEvery
             ), frequency(1.0 / timeConstant_),
                kineticEnergy(GPUArrayGlobal<float>(2)),
                ke_current(0.0), ndf(0),
                chainLength(3), nTimesteps(1), n_ys(1),
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
                tempComputer(state, true, false), 
                pressComputer(state, true, false), 
                pressMode(PRESSMODE::ISO),
                thermostatting(true),
                barostatting(false)
{
    pressComputer.usingExternalTemperature = true;
}

FixNoseHoover::FixNoseHoover(boost::shared_ptr<State> state_, std::string handle_,
                             std::string groupHandle_, py::object tempFunc_, double timeConstant_)
        : tempInterpolator(tempFunc_),
          Fix(state_,
              handle_,           // Fix handle
              groupHandle_,      // Group handle
              NoseHooverType,   // Fix name
              false,            // forceSingle
              false,            // requiresVirials 
              false,            // requiresCharges
              1                 // applyEvery
             ), frequency(1.0 / timeConstant_),
                kineticEnergy(GPUArrayGlobal<float>(2)),
                ke_current(0.0), ndf(0),
                chainLength(3), nTimesteps(1), n_ys(1),
                weight(std::vector<double>(n_ys,1.0)),
                //thermPos(std::vector<double>(chainLength,0.0)),
                thermVel(std::vector<double>(chainLength,0.0)),
                thermForce(std::vector<double>(chainLength,0.0)),
                thermMass(std::vector<double>(chainLength,0.0)),
                omega(std::vector<double>(6)),
                omegaVel(std::vector<double>(6)),
                omegaMass(std::vector<double>(6)),
                pressFreq(6, 0),
                pressCurrent(6, 0),
                pFlags(6, false),
                scale(make_float3(1.0f, 1.0f, 1.0f)),
                tempComputer(state, true, false), 
                pressComputer(state, true, false), 
                pressMode(PRESSMODE::ISO),
                thermostatting(true),
                barostatting(true)

{
    pressComputer.usingExternalTemperature = true;
}


FixNoseHoover::FixNoseHoover(boost::shared_ptr<State> state_, std::string handle_,
                             std::string groupHandle_, py::list intervals_, py::list temps_, double timeConstant_)
        : tempInterpolator(intervals_, temps_),
          Fix(state_,
              handle_,           // Fix handle
              groupHandle_,      // Group handle
              NoseHooverType,   // Fix name
              false,            // forceSingle
              false,            // requiresVirials 
              false,            // requiresCharges
              1                 // applyEvery
             ), frequency(1.0 / timeConstant_),
                kineticEnergy(GPUArrayGlobal<float>(2)),
                ke_current(0.0), ndf(0),
                chainLength(3), nTimesteps(1), n_ys(1),
                weight(std::vector<double>(n_ys,1.0)),
                //thermPos(std::vector<double>(chainLength,0.0)),
                thermVel(std::vector<double>(chainLength,0.0)),
                thermForce(std::vector<double>(chainLength,0.0)),
                thermMass(std::vector<double>(chainLength,0.0)),
                omega(std::vector<double>(6)),
                omegaVel(std::vector<double>(6)),
                omegaMass(std::vector<double>(6)),
                pressFreq(6, 0),
                pressCurrent(6, 0),
                pFlags(6, false),
                scale(make_float3(1.0f, 1.0f, 1.0f)),
                tempComputer(state, true, false), 
                pressComputer(state, true, false), 
                pressMode(PRESSMODE::ISO),
                thermostatting(true),
                barostatting(true)

{
    pressComputer.usingExternalTemperature = true;
}
void FixNoseHoover::setPressure(double press) {
    pressInterpolator = Interpolator(press);
}


bool FixNoseHoover::prepareForRun()
{
    // Calculate current kinetic energy
    tempInterpolator.turnBeginRun = state->runInit;
    tempInterpolator.turnFinishRun = state->runInit + state->runningFor;
    tempComputer.prepareForRun();
    if (barostatting) {
        pressComputer.prepareForRun();
    }

    calculateKineticEnergy();
    tempInterpolator.computeCurrentVal(state->runInit);
    updateMasses();

    // Update thermostat forces
    double boltz = 1.0;
    double temp = tempInterpolator.getCurrentVal();
    thermForce.at(0) = (ke_current - ndf * boltz * temp) / thermMass.at(0);
    for (size_t k = 1; k < chainLength; ++k) {
        thermForce.at(k) = (
                thermMass.at(k-1) *
                thermVel.at(k-1) *
                thermVel.at(k-1) - boltz*temp
            ) / thermMass.at(k);
    }
    if (barostatting) {
        for (int i=0; i<6; i++) {
            omega[i] = 0;
            omegaVel[i] = 0;
            omegaMass[i] = temp * boltz * state->atoms.size() / (pressFreq[i] * pressFreq[i]); //
        }
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
    }

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

void FixNoseHoover::omegaIntegrate() {
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

    //! \todo Until now, we assume Boltzmann-constant = 1.0. Consider allowing
    //!       other units.
    double boltz = 1.0;

    // Update the desired temperature
    double temp;
    if (firstHalfStep) {
        double currentTemp = tempInterpolator.getCurrentVal();
        tempInterpolator.computeCurrentVal(state->turn);
        temp = tempInterpolator.getCurrentVal();
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
    double boltz = 1.0;
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
        //tempComputer.tempScalar;

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
