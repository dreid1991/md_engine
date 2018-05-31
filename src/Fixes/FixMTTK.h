#pragma once
#ifndef FIXMTTK_H
#define FIXMTTK_H

#include "Fix.h"
#include "GPUArrayGlobal.h"

#include <vector>

#include <boost/shared_ptr.hpp>

#include "Interpolator.h"
#include "DataComputerTemperature.h"
#include "DataComputerPressure.h"
namespace py = boost::python;

//! Make FixMTTK available to the python interface
void export_FixMTTK();

class FixMTTK : public Fix {
public:
    //! Delete default constructor
    FixMTTK() = delete;

    //! Constructor
    /*!
     * \param state Pointer to the simulation state
     * \param handle "Name" of the Fix
     * \param groupHandle String specifying group of atoms this Fix acts on
     */
    FixMTTK(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle);

    //! Prepare Nose-Hoover thermostat for simulation run
    bool prepareFinal();
    //bool prepareForRun();

    //! Perform post-Run operations
    bool postRun();

    //! First half step of the integration
    /*!
     * \return Result of the FixMTTK::halfStep() call.
     */
    bool stepInit();

    //! Second half step of the integration
    /*!
     * \return Result of FixMTTK::halfStep() call.
     */
    bool stepFinal();

    bool postNVE_V();

    bool postNVE_X();
    //! Set the pressure, time constant, and pressure mode for this barostat
    /*!
     * \param pressFunc The pressure set point, as a python function
     * \param timeconstant The time constant associated with this barostat
     */
    
    // --- for mode, we only do isotropic.
    void setPressure(py::object, double);
    //! Set the pressure, time constant, and pressure mode for this barostat
    /*!
     * \param press The pressure set point
     * \param timeconstant The time constant associated with this barostat
     */
    void setPressure(double, double);
    
    //! Set the pressure, time constant, and pressure mode for this barostat
    /*!
     * \param pressures The pressure set point, as a list of intervals
     * \param intervals The time intervals for which the above list of pressures constitute the set point
     * \param timeconstant The time constant associated with this barostat
     */
    void setPressure(py::list, py::list, double);

    //! Set the temperature set point and associated time constant for this thermostat
    /*!
     * \param tempFunc The set point temperature, as a python function
     * \param timeConstant The time constant associated with this thermostat
     */
    void setTemperature(py::object, double);

    //! Set the temperature set point and associated time constant for this thermostat
    /*!
     * \param temperature The set point temperature
     * \param timeConstant The time constant associated with this thermostat
     */
    void setTemperature(double, double);
    
    //! Set the temperature set point and associated time constant for this thermostat
    /*!
     * \param temps The set point temperature as a list of temperature set points
     * \param intervals The list of timestep intervals for the list of temperature set points
     * \param timeConstant The time constant associated with this thermostat
     */
    void setTemperature(py::list, py::list, double);
    
    double compressibility;
    std::vector<double> compressibility_matrix; // as matrix
   
private:
    
    // boolean, are we barostatting?
    bool barostatting;

    bool iterative;

    void initial_iteration();

    GPUArrayDeviceGlobal<real4> xs_copy;   //!< For storing copies of device array
    GPUArrayDeviceGlobal<real4> vs_copy;   //!< For storing copies of device array
    GPUArrayDeviceGlobal<Virial> virials_copy; //!< For storing copies of device array
    Virial virials_sum;
    Virial virials_sum_old;
    Virial virials_from_constraints_old;
    Virial virials_from_constraints;

    double vol0;
    double MassQ_Winv;
    
    uint32_t groupTag;

    std::vector<double> MassQ_Winvm;
    std::vector<double> MassQ_Qinv;
    double bmass;
    std::vector<double> MassQ_QPinv;

    std::vector<double> xi;
    std::vector<double> vxi;
    
    double veta;
    double vetanew;
    double alpha;
    double GW;

    std::vector<double> press_xi;
    std::vector<double> press_vxi;
    double thermalIntegral;

    double rscale;
    double vscale;
    double vrscale;

    // fixed suzuki-yoshida order of 5;
    // this should not ever be changed.
    static const int sy_order = 5;

    double setPointTemperature;
    double oldSetPointTemperature;
    double setPointPressure;
    double oldSetPointPressure;
    double DIMENSIONS;

    double traceKE; // trace of the kinetic energy tensor from temperature computer
    int nh_chainlength;
    double nhc_scale; // as ekinscalef_nhc
    double scalefac;
    double vscale_nhc; // as vscale_nhc;
    int ndf;

    double tFrequency;  // thermostat frequency
    double pFrequency;  // barostat frequency

    Virial KE_tensor; // Virial data type, convenient representation of tensor
    void trotter_integrate_barostat();
    void trotter_integrate_thermostat();
    void apply_nhc_scale();
    void trotter_integrate_boxv();
    void calculateKineticEnergy();
    void updateThermalMasses();
    void updateBarostatThermalMasses();


    // find all instances of rigid fixes in the simulation
    std::vector<std::shared_ptr<Fix *> > constraint_fixes;

    Interpolator *getInterpolator(std::string);

    Interpolator tempInterpolator;
    Interpolator pressInterpolator;

    MD_ENGINE::DataComputerTemperature tempComputer;
    MD_ENGINE::DataComputerPressure pressComputer;

};

#endif
