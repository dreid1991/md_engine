#pragma once
#ifndef FIXNOSEHOOVER_H
#define FIXNOSEHOOVER_H

#include "Fix.h"
#include "GPUArrayGlobal.h"

#include <vector>

#include <boost/shared_ptr.hpp>

//! Make FixNoseHoover available to the python interface
void export_FixNoseHoover();

//! Nose-Hoover Thermostat and Barostat
/*!
 * Fix to sample the canonical ensemble of a given system. Implements the
 * popular Nose-Hoover thermostat and barostat.
 *
 * The implementation is based on the algorithm proposed by Martyna et al.
 * \cite MartynaEtal:MP1996 .
 *
 * \todo Implement barostat.
 */
class FixNoseHoover : public Fix {
public:
    //! Delete default constructor
    FixNoseHoover() = delete;

    //! Constructor
    /*!
     * \param state Pointer to the simulation state
     * \param handle "Name" of the Fix
     * \param groupHandle String specifying group of atoms this Fix acts on
     * \param temp Desired temperature of the system
     * \param timeConstant Time constant of the Nose-Hoover thermostat
     */
    FixNoseHoover(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle,
                  float temp,
                  float timeConstant);

    //! First half step of the integration
    /*!
     * \return Result of the FixNoseHoover::halfStep() call.
     */
    virtual bool stepInit();

    //! Second half step of the integration
    /*!
     * \return Result of FixNoseHoover::halfStep() call.
     */
    virtual bool stepFinal();

private:
    //! Perform one half step of the Nose-Hoover thermostatting
    /*!
     * \return False if a problem occured, else return True
     */
    bool halfStep();

    //! Get the total kinetic energy
    /*!
     * \return Total kinetic energy
     *
     * Calculate the total kinetic energy of the atoms in the Fix group
     */
    void calculateKineticEnergy();

    //! Rescale particle velocities
    /*!
     * \param scale Scale factor for rescaling
     */
    void rescale(float scale);

    double temp; //!< Desired temperature
    GPUArrayGlobal<float> kineticEnergy; //!< Stores kinetic energy and
                                         //!< number of atoms in Fix group

    size_t chainLength; //!< Number of thermostats in the Nose-Hoover chain
    size_t nTimesteps; //!< Number of timesteps for multi-timestep method

    size_t n_ys; //!< n_ys from \cite MartynaEtal:MP1996
    std::vector<double> weight; //!< Weights for closer approximation

    std::vector<double> thermPos; //!< Position of the Nose-Hoover thermostats
    std::vector<double> thermVel; //!< Velocity of the Nose-Hoover thermostats
    std::vector<double> thermForce; //!< Force on the Nose-Hoover thermostats
    std::vector<double> thermMass; //!< Masses of the Nose-Hoover thermostats
};

#endif
