#pragma once
#ifndef FIXNOSEHOOVER_H
#define FIXNOSEHOOVER_H

#include "Fix.h"
#include "GPUArrayGlobal.h"

#include <vector>

#include <boost/shared_ptr.hpp>

#include "Interpolator.h"
#include "DataComputerTemperature.h"
#include "DataComputerPressure.h"

//! Make FixNoseHoover available to the python interface
void export_FixNoseHoover();

//! Nose-Hoover Thermostat and Barostat
/*!
 * Fix to sample the canonical ensemble of a given system. Implements the
 * popular Nose-Hoover thermostat and barostat.
 *
 *
 * The implementation is based on the algorithm proposed by Martyna et al.
 * \cite MartynaEtal:MP1996 .
 *
 * The barostat implementation is based off of Martyna et. al.,
 * J. Phys. A: Math. Gen. 39 (2006) 5629-5651.
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

    void calculateKineticEnergy();
    //! This function updates the thermostat masses
    /*!
     * The masses of the thermostat depend on the desired temperature. Thus,
     * they should be updated when the desired temperature is changed.
     *
     * This function assumes that ke_current and ndf have already been
     * calculated and are up to date.
     */
    void updateThermalMasses();

    //! This function gets the instantaneous pressure from the pressure computer 
    /*!
     * The instantaneous pressure is required for barostatting.
     * It should be updated prior integrating the barostat momenta.
     *
     * This function assumes that the virials and temperature are up to date.
     * This is also where partitioning of the internal stress tensor occurs - 
     * e.g., coupling of the XYZ dimensions, or no coupling.
     */
    void getCurrentPressure();


    //! This function updates the mass variables associated with the barostat/cell parameters
    /*!
     * The mass variables associated with the barostat/cell parameters are functions
     * of the externally imposed temperature, and thus vary as the set point varies.
     *
     * This function uses the set point temperature and the user-specified frequencies
     * to compute the masses of the barostat/cell parameters.
     */
    void updateBarostatMasses(bool);

    //! This function updates the mass variables associated with the barostat thermostats
    /*!
     * See 'updateThermalMasses' documentation above.
     */
    void updateBarostatThermalMasses(bool);

    //! This function integrates the thermostat variables for the barostats
    void barostatThermostatIntegrate(bool);



    Interpolator *getInterpolator(std::string);
    //! Rescale particle velocities
    /*!
     * \param scale Scale factor for rescaling
     */
    void rescale();
    void setPressure(double pressure);

    float frequency; //!< Frequency of the Nose-Hoover thermostats
    Virial pFreq; //!< Frequency of the Nose-Hoover barostats
    GPUArrayGlobal<float> kineticEnergy; //!< Stores kinetic energy and
                                         //!< number of atoms in Fix group
    float ke_current; //!< Current kinetic energy
    size_t ndf; //!< Number of degrees of freedom

    Virial currentPressure; //!< Current pressure, with (or without) coupling
    Virial setPointPressure; //!< Our current set point pressure, from pressInterpolator
    Virial setPointTemperature; //!< Our current set point temperature, from tempInterpolator
    Virial oldSetPointPressure; //!< The set point pressure from the previous turn
    Virial oldSetPointTemperature; //!< The set point temperature from the previous turn

    size_t chainLength; //!< Number of thermostats in the Nose-Hoover chain
    size_t nTimesteps; //!< Number of timesteps for multi-timestep method

    size_t n_ys; //!< n_ys from \cite MartynaEtal:MP1996
    std::vector<double> weight; //!< Weights for closer approximation

    //std::vector<double> thermPos; //!< Position (= Energy) of the Nose-Hoover
                                    //!< thermostats
    std::vector<double> thermVel; //!< Velocity of the Nose-Hoover thermostats
    std::vector<double> thermForce; //!< Force on the Nose-Hoover thermostats
    std::vector<double> thermMass; //!< Masses of the Nose-Hoover thermostats

    // note to others: we use Virial class for convenience for any vector of floats 
    // that is required to have 6 and only 6 values
    // -- allows for convenient mathematical operations via class-defined operators
    Virial pressMass; //!< Masses of the Nose-Hoover barostats
    std::vector<double> pressThermMass; //!< Masses of the Nose-Hoover barostats' thermostats
    std::vector<double> pressThermVel; //!< Velocity of the Nose-Hoover barostats' thermostats
    std::vector<double> pressThermForce; //!< Force on the Nose-Hoover barostats' thermostats

    double boltz; //!< Local copy of our boltzmann constant with proper units
    float3 epsilon; //!< Epsilon, ratio of V/V0; alternatively, our volume scaling parameter
    int nAtoms; //!< Number of atoms, N, in the system

    Virial identity; //!< identity tensor made using our Virial class (6-value vector)

    int nDimsBarostatted; //!< The number of dimensions that are barostatted

    std::vector<double> pressFreq;
    void thermostatIntegrate(double, double, bool);
    void omegaIntegrate();
    void scaleVelocitiesOmega();
    std::vector<bool> pFlags;
    Interpolator tempInterpolator;
    Interpolator pressInterpolator;

    float3 scale; //!< Factor by which the velocities are rescaled
    MD_ENGINE::DataComputerTemperature tempComputer;
    MD_ENGINE::DataComputerPressure pressComputer;
    bool thermostatting;
    bool barostatting;
    int pressMode;

};

#endif
