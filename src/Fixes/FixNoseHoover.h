#pragma once
#ifndef FIXNOSEHOOVER_H
#define FIXNOSEHOOVER_H

#include "Fix.h"
#include "GPUArrayGlobal.h"

#include <vector>

#include <boost/shared_ptr.hpp>

#include "FixThermostatBase.h"

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
 * Note that the Nose-Hoover thermostat should only be used with the
 * IntegratorVerlet.
 *
 * \todo Allow to specify desired length of Nose-Hoover chain
 * \todo Allow to set multiple-timestep integration
 * \todo Allow to use higher-order approximations
 *
 * \todo Implement barostat.
 */
class FixNoseHoover : public FixThermostatBase, public Fix {
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
                  double temp,
                  double timeConstant);
    FixNoseHoover(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle,
                  boost::python::list intervals,
                  boost::python::list temps,
                  double timeConstant);
    FixNoseHoover(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle,
                  boost::python::object tempFunc,
                  double timeConstant);

    //! Prepare Nose-Hoover thermostat for simulation run
    bool prepareForRun();

    //! Perform post-Run operations
    bool postRun();

    //! First half step of the integration
    /*!
     * \return Result of the FixNoseHoover::halfStep() call.
     */
    bool stepInit();

    //! Second half step of the integration
    /*!
     * \return Result of FixNoseHoover::halfStep() call.
     */
    bool stepFinal();

private:
    //! Perform one half step of the Nose-Hoover thermostatting
    /*!
     * \param firstHalfStep True for the first half step of the integration
     *
     * \return False if a problem occured, else return True
     */
    bool halfStep(bool firstHalfStep);

    //! This function updates the desired temperature
    /*!
     * The thermostat temperature can change during the course of the
     * simulation. The temperature depends on the timestep and is updated in
     * this function.
     */
    bool updateTemperature();

    //! This function updates the thermostat masses
    /*!
     * The masses of the thermostat depend on the desired temperature. Thus,
     * they should be updated when the desired temperature is changed.
     *
     * This function assumes that ke_current and ndf have already been
     * calculated and are up to date.
     */
    void updateMasses();

    //! Get the total kinetic energy
    /*!
     * Calculate the total kinetic energy of the atoms in the Fix group
     */
    void calculateKineticEnergy();

    //! Rescale particle velocities
    /*!
     * \param scale Scale factor for rescaling
     */
    void rescale();

    float frequency; //!< Frequency of the Nose-Hoover thermostats

    GPUArrayGlobal<float> kineticEnergy; //!< Stores kinetic energy and
                                         //!< number of atoms in Fix group
    float ke_current; //!< Current kinetic energy
    size_t ndf; //!< Number of degrees of freedom

    size_t chainLength; //!< Number of thermostats in the Nose-Hoover chain
    size_t nTimesteps; //!< Number of timesteps for multi-timestep method

    size_t n_ys; //!< n_ys from \cite MartynaEtal:MP1996
    std::vector<double> weight; //!< Weights for closer approximation

    //std::vector<double> thermPos; //!< Position (= Energy) of the Nose-Hoover
                                    //!< thermostats
    std::vector<double> thermVel; //!< Velocity of the Nose-Hoover thermostats
    std::vector<double> thermForce; //!< Force on the Nose-Hoover thermostats
    std::vector<double> thermMass; //!< Masses of the Nose-Hoover thermostats

    float scale; //!< Factor by which the velocities are rescaled
};

#endif
