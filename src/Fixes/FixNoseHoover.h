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
class FixNoseHoover : public Fix {
public:
    //! Delete default constructor
    FixNoseHoover() = delete;

    //! Constructor
    /*!
     * \param state Pointer to the simulation state
     * \param handle "Name" of the Fix
     * \param groupHandle String specifying group of atoms this Fix acts on
     * \param timeConstant Time constant of the Nose-Hoover thermostat
     */

    // general constructor for barostat/thermostat;
    FixNoseHoover(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle,
                  double timeConstant);

    // declare set methods for the thermostat
    // assorted inputs are accepted for set point temperatures and pressures
    void setTemperature(double);
    void setTemperature(boost::python::object);
    void setTemperature(boost::python::list, boost::python::list);
    
    void setPressure(double);
    void setPressure(boost::python::object);
    void setPressure(boost::python::list, boost::python::list);

    // likewise, we offer default values for chain lengths;
    // conversely, these lengths may be set, /provided/ they are >= 1
    void setBarostatChainLength(int);
    void setBarostatThermostatChainLength(int);



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
    void updateMasses();

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
    size_t pchainLength; //!< Number of thermostats monitoring the barostat's thermal DOF
    size_t nTimesteps; //!< Number of timesteps for multi-timestep method

    size_t n_ys; //!< n_ys from \cite MartynaEtal:MP1996
    std::vector<double> weight; //!< Weights for closer approximation

    //std::vector<double> thermPos; //!< Position (= Energy) of the Nose-Hoover
                                    //!< thermostats
    std::vector<double> thermVel; //!< Velocity of the Nose-Hoover thermostats
    std::vector<double> thermForce; //!< Force on the Nose-Hoover thermostats
    std::vector<double> thermMass; //!< Masses of the Nose-Hoover thermostats

    // our barostat variables
    std::vector<double> omega;
    std::vector<double> omegaVel;
    std::vector<double> omegaMass;

    int etaPChainLength; //!< length of barostat-thermostat chain; default value 3
    std::vector<double> etaPressure; //!< Position of the barostat-thermostats
    std::vector<double> etaPressure_dt; //!< Velocities of the barostat-thermostats
    std::vector<double> etaPressure_dt2; //!< Accelerations of the barostat-thermostats
    std::vector<double> etaPressure_mass; //!< Masses of the barostat-thermostats



    //TODO: verify that we have these quantities; also, where would we get them from? Bounds.h?
    std::vector<double> h0; //!< Our reference cell 
    std::vector<double> h0_inv; //!< inverse;
    std::vector<double> referencePoint; //!< reference point within the original cell
    void tranformBox(); //TODO: modifying the actual simulation box, be very careful
    
    
    
    
    bool verifyInputs(); //TODO: this will be called in prepareForRun, verifying that the barostat is
    // exactly specified;



    std::vector<double> pressFreq;
    std::vector<double> pressCurrent;
    void setPressCurrent();
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

    float mtkTerm1;
    float mtkTerm2;

    // flags for bookkeeping
    bool barostatThermostatChainLengthSpecified;











};

#endif
