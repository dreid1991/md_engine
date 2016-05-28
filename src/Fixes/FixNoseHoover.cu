#include "FixNoseHoover.h"

#include <cmath>
#include <string>

#include <boost/python.hpp>

#include "cutils_func.h"
#include "Logging.h"
#include "State.h"

namespace py = boost::python;

std::string NoseHooverType = "NoseHoover";

// CUDA function to calculate the total kinetic energy

// CUDA function to rescale particle velocities
__global__ void rescale_cu(int nAtoms, uint groupTag, float4 *vs, float4 *fs, float scale)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint groupTagAtom = ((uint *) (fs+idx))[3];
        if (groupTag & groupTagAtom) {
            float4 vel = vs[idx];
            float invmass = vel.w;
            vel *= scale;
            vel.w = invmass;
            vs[idx] = vel;
        }
    }
}
__global__ void rescale_no_tags_cu(int nAtoms, float4 *vs, float scale)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 vel = vs[idx];
        float invmass = vel.w;
        vel *= scale;
        vel.w = invmass;
        vs[idx] = vel;
    }
}

FixNoseHoover::FixNoseHoover(boost::shared_ptr<State> state, std::string handle,
                             std::string groupHandle, float temp, float timeConstant)
        : Fix(state,
              handle,           // Fix handle
              groupHandle,      // Group handle
              NoseHooverType,   // Fix name
              false,            // forceSingle
              false,            // requiresVirials 
              false,            // requiresCharges
              1                 // applyEvery
             ), temp(temp), frequency(1.0 / timeConstant),
                kineticEnergy(GPUArrayGlobal<float>(2)),
                ke_current(0.0), ndf(0),
                chainLength(3), nTimesteps(1), n_ys(1),
                weight(std::vector<double>(n_ys,1.0)),
                //thermPos(std::vector<double>(chainLength,0.0)),
                thermVel(std::vector<double>(chainLength,0.0)),
                thermForce(std::vector<double>(chainLength,0.0)),
                thermMass(std::vector<double>(chainLength,0.0)),
                scale(1.0f)
{

}

bool FixNoseHoover::prepareForRun()
{
    // Calculate current kinetic energy
    calculateKineticEnergy();
    updateMasses();

    // Update thermostat forces
    double boltz = 1.0;
    thermForce.at(0) = (ke_current - ndf * boltz * temp) / thermMass.at(0);
    for (size_t k = 1; k < chainLength; ++k) {
        thermForce.at(k) = (
                thermMass.at(k-1) *
                thermVel.at(k-1) *
                thermVel.at(k-1) - boltz*temp
            ) / thermMass.at(k);
    }

    return true;
}

bool FixNoseHoover::postRun()
{
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
    if (firstHalfStep) {
        if (updateTemperature()) {
            updateMasses();
        }
    }

    // Get the total kinetic energy
    if (!firstHalfStep) {
        //! \todo This optimization assumes that the velocities are not changed
        //!       between stepFinal() and stepInit(). Can we add a check to make
        //!       sure this is indeed the case?
        calculateKineticEnergy();
    }

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

    // Update particle velocites
    //! \todo In this optimization, I assume that velocities are not accessed
    //!       between stepFinal() and stepInitial(). Can we ensure that this is
    //!       indeed the case?
    if (firstHalfStep) {
        rescale();
    }

    return true;
}

bool FixNoseHoover::updateTemperature()
{
    // This should be modified to allow for temperature changes
    double newTemp = temp;

    if (temp != newTemp) {
        // Temperature changed
        temp = newTemp;
        return true;
    }

    // Temperature remained unchanged
    return false;
}

void FixNoseHoover::updateMasses()
{
    double boltz = 1.0;

    thermMass.at(0) = ndf * boltz * temp / (frequency*frequency);
    for (size_t i = 1; i < chainLength; ++i) {
        thermMass.at(i) = boltz*temp / (frequency*frequency);
    }
}


void FixNoseHoover::calculateKineticEnergy()
{
    size_t nAtoms = state->atoms.size();
    kineticEnergy.d_data.memset(0);
    if (groupTag == 1) { //if is all atoms
       sumVectorSqr3DOverW<float, float4, 2>
            <<<NBLOCK(nAtoms / (double) 2), PERBLOCK, 2*PERBLOCK*sizeof(float)>>>(
                    kineticEnergy.d_data.data(),
                    state->gpd.vs.getDevData(),
                    nAtoms,
                    state->devManager.prop.warpSize
            );
       //std::cout << " calling tag-less sum " << std::endl;
       kineticEnergy.dataToHost();
       cudaDeviceSynchronize();
       ndf = state->atoms.size();

    } else {
        sumVectorSqr3DTagsOverW<float, float4, 2>
            <<<NBLOCK(nAtoms / (double) 2), PERBLOCK, 2*PERBLOCK*sizeof(float)>>>(
                    kineticEnergy.d_data.data(),
                    state->gpd.vs.getDevData(),
                    nAtoms,
                    groupTag,
                    state->gpd.fs.getDevData(),
                    state->devManager.prop.warpSize
            );
        kineticEnergy.dataToHost();
        cudaDeviceSynchronize();
        ndf = *((int *) (kineticEnergy.h_data.data()+1));
    }

    ke_current = kineticEnergy.h_data[0];
  //  std::cout << "current " << ke_current << std::endl;
    if (state->is2d) {
        ndf *= 2;
    } else {
        ndf *= 3;
    }
//    std::cout << "temp is " << ke_current / ndf << std::endl;
}

void FixNoseHoover::rescale()
{
    if (scale == 1.0f) {
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

    scale = 1.0f;
}

void export_FixNoseHoover()
{
    py::class_<FixNoseHoover,                    // Class
               boost::shared_ptr<FixNoseHoover>, // HeldType
               py::bases<Fix>,                   // Base class
               boost::noncopyable>
    (
        "FixNoseHoover",
        py::init<boost::shared_ptr<State>, std::string, std::string, float, float>(
            py::args("state", "handle", "groupHandle", "temp", "timeConstant")
        )
    )
    ;
}