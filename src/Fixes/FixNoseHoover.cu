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

FixNoseHoover::FixNoseHoover(boost::shared_ptr<State> state, std::string handle,
                             std::string groupHandle, float temp, float timeConstant)
        : Fix(state,
              handle,           // Fix handle
              groupHandle,      // Group handle
              NoseHooverType,   // Fix name
              false,            // forceSingle
              1                 // applyEvery
             ), temp(temp), kineticEnergy(GPUArrayGlobal<float>(2)),
                chainLength(3), nTimesteps(1), n_ys(1),
                weight(std::vector<double>(n_ys,1.0)),
                thermPos(std::vector<double>(chainLength,0.0)),
                thermVel(std::vector<double>(chainLength,0.0)),
                thermForce(std::vector<double>(chainLength,0.0)),
                thermMass(std::vector<double>(chainLength))
{
    //! \todo Consider using other units
    double boltz = 1.0;

    //! \todo How to get number of degrees of freedom here?
    int ndf = state->atoms.size();
    if (state->is2d) {
        ndf *= 2;
    } else {
        ndf *= 3;
    }

    double freq = 1.0 / timeConstant;
    thermMass.at(0) = ndf*boltz*temp / (freq*freq);
    for (size_t i = 1; i < chainLength; ++i) {
        thermMass.at(i) = boltz*temp / (freq*freq);
    }
}

bool FixNoseHoover::stepInit()
{
    return halfStep();
}

bool FixNoseHoover::stepFinal()
{
    return halfStep();
}

bool FixNoseHoover::halfStep()
{
    if (chainLength == 0) {
        mdWarning("Call of FixNoseHoover with zero thermostats in "
                  "the Nose-Hoover chain.");
        return false;
    }

    //! \todo Until now, we assume Boltzmann-constant = 1.0. Consider allowing
    //!       other units.
    double boltz = 1.0;

    double scale = 1.0;

    // Get the total kinetic energy
    calculateKineticEnergy();
    float kinEnergy = kineticEnergy.h_data[0];

    // Get number of degrees of freedom
    size_t ndf = *((int *) (kineticEnergy.h_data.data()+1));
    if (state->is2d) {
        ndf *= 2;
    } else {
        ndf *= 3;
    }

    // Equipartition at desired temperature
    double nkt = ndf * boltz * temp;

    // Update the forces
    thermForce.at(0) = (kinEnergy - nkt) / thermMass.at(0);

    // Multiple timestep procedure
    for (size_t i = 0; i < nTimesteps; ++i) {
        for (size_t j = 0; j < n_ys; ++j) {
            double timestep = weight.at(j)*state->dt / nTimesteps;

            // Update thermostat forces
            //! \todo Consider shorter thermostat chains
            thermForce.at(chainLength-1) =
                (
                    thermMass.at(chainLength-2) *
                    thermVel.at(chainLength-2) *
                    thermVel.at(chainLength-2) - boltz*temp
                ) / thermMass.at(chainLength-1);

            // Update thermostat velocities
            thermVel.back() += 0.25*timestep*thermForce.back();
            for (size_t k = chainLength-2; k > 0; --k) {
                double preFactor = std::exp( -0.125*timestep*thermVel.at(k+1) );
                thermVel.at(k) *= preFactor;
                thermForce.at(k) = (
                        thermMass.at(k-1) *
                        thermVel.at(k-1) *
                        thermVel.at(k-1) - boltz*temp
                    ) / thermMass.at(k);
                thermVel.at(k) += 0.25*timestep * thermForce.at(k);
                thermVel.at(k) *= preFactor;
            }

            double preFactor = std::exp( -0.125*timestep*thermVel.at(1) );
            thermVel.at(0) *= preFactor;
            thermVel.at(0) += 0.25*timestep*thermForce.at(0);
            thermVel.at(0) *= preFactor;

            // Update particle velocities
            double scaleFactor = std::exp( -0.5*timestep*thermVel.at(0) );
            scale *= scaleFactor;

            kinEnergy *= scaleFactor*scaleFactor;

            // Update the thermostat positions
            for (size_t k = 0; k < chainLength; ++k) {
                thermPos.at(k) += 0.5*timestep*thermVel.at(k);
            }

            // Update the forces
            preFactor = std::exp( -0.125*timestep*thermVel.at(1) );
            thermVel.at(0) *= preFactor;
            thermForce.at(0) = (kinEnergy - nkt) / thermMass.at(0);
            thermVel.at(0) += 0.25*timestep * thermForce.at(0);
            thermVel.at(0) *= preFactor;

            // Update thermostat velocities
            for (size_t k = 1; k < chainLength-1; ++k) {
                preFactor = std::exp( -0.125*timestep*thermVel.at(k+1) );
                thermVel.at(k) *= preFactor;
                thermForce.at(k) = (
                        thermMass.at(k-1) *
                        thermVel.at(k-1) *
                        thermVel.at(k-1) - boltz*temp
                    ) / thermMass.at(k);
                thermVel.at(k) += 0.25*timestep * thermForce.at(k);
                thermVel.at(k) *= preFactor;
            }

            thermForce.at(chainLength-1) = (
                    thermMass.at(chainLength-2) *
                    thermVel.at(chainLength-2) *
                    thermVel.at(chainLength-2) - boltz*temp
                ) / thermMass.at(chainLength-1);
            thermVel.at(chainLength-1) += 0.25*timestep*thermForce.at(chainLength-1);
        }
    }

    // Update particle velocites
    // scale gets converted to float, losing precision
    rescale(scale);

    return true;
}

void FixNoseHoover::calculateKineticEnergy()
{
    size_t nAtoms = state->atoms.size();
    kineticEnergy.d_data.memset(0);
    SAFECALL((sumVectorSqr3DTagsOverW<float, float4>
        <<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)>>>(
                kineticEnergy.d_data.data(),
                state->gpd.vs.getDevData(),
                nAtoms,
                groupTag,
                state->gpd.fs.getDevData(),
                state->devManager.prop.warpSize
        )));
    kineticEnergy.dataToHost();
    cudaDeviceSynchronize();
}

void FixNoseHoover::rescale(float scale)
{
    size_t nAtoms = state->atoms.size();
    rescale_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms,
                                             groupTag,
                                             state->gpd.vs.getDevData(),
                                             state->gpd.fs.getDevData(),
                                             scale);
}

void export_FixNoseHoover()
{
    py::class_<FixNoseHoover,                    // Class
               boost::shared_ptr<FixNoseHoover>, // HeldType
               py::bases<Fix>,                   // Base class
               boost::noncopyable>
    (
        "FixNoseHoover",
        py::init<boost::shared_ptr<State>, std::string, std::string, float>(
            py::args("state", "handle", "groupHandle", "temp")
        )
    )
    ;
}
