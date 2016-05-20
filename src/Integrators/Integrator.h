#pragma once
#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <boost/python/list.hpp>
#include <string>
#include <vector>

class GPUArray;
class State;

void export_Integrator();

//! Base class for Molecular Dynamics Integrators
/*
 * This class is a base class for all MD Integrators. It takes care of all
 * aspects common to all integrators such as doing basic checks, data transfer
 * from and to the GPU, etc.
 */
class Integrator {

protected:
    //! Calculate force for all fixes
    /*!
     * \param computeVirials Compute virials for all forces if True
     *
     * This function iterates over all fixes and if the Fix should be applied
     * its force (and virials) is computed.
     */
    void force(bool computeVirials);
    
    //! Perform all asynchronous operations
    /*!
     * This function performs all asynchronous operations, such as writing
     * configurations or performing Python operations
     */
    void asyncOperations();
    std::vector<GPUArray *> activeData; //!< List of pointers to the data
                                        //!< used by this integrator

    //! Simple checks before the run
    /*!
     * The checks consist of:
     *   - GPU device compatibility needs to be >= 3.0
     *   - Atom grid needs to be set
     *   - Cutoff distance needs to be set
     *   - 2d system may not be periodic in z dimension
     *   - Grid discretization must not be smaller than cutoff distance plus
     *     padding
     */
    void basicPreRunChecks();

    //! Prepare Integrator for running
    /*!
     * \param numTurns Number of turns the integrator is expected to run
     *
     * Prepare the integrator to run for a given amount of timesteps. This
     * includes copying all data to the GPU device and calling prepareForRun()
     * on all fixes.
     */
    void basicPrepare(int numTurns);

    //! Finish simulation run
    /*!
     * Finish the simulation run. This includes copying all relevant data to
     * the CPU host and calling postRun on all fixes.
     */
    void basicFinish();

    //! Collect all pointers to the relevant data into activeData
    void setActiveData();

    //! Collect data for all DataSets
    void doDataCollection();

    //! Calculate single point energy for all fixes
    /*!
     * A single point energy excludes energy/forces from thermostat fixes and
     * the likes.
     */
    void singlePointEng(); //! \todo make a python-wrapped version

public:
    //! Calculate and return single point energy
    /*!
     * \param groupHandle Handle defining the group used for averaging
     *
     * \return Average energy for all particles in the specified group
     *
     * This function calculates and returns the average per particle energy for
     * the particles in the group specified via the groupHandle.
     */
    double singlePointEngPythonAvg(std::string groupHandle);

    //! Create list of per-particle energies
    /*!
     * \return List containing the per-particle energy for each atom
     *
     * This function calculates the per-particle energy and returns a list
     * containing one value per atom in the simulation.
     */
    boost::python::list singlePointEngPythonPerParticle();
    State *state; //!< Pointer to the simulation state

    //! Default constructor
    /*!
     * \todo Do we need a default constructor?
     */
    Integrator() {};

    //! Constructor
    /*!
     * \param state_ Pointer to simulation state
     * \param type_ String specifying type of Integrator (unused)
     *
     * \todo Remove usage of type
     */
    explicit Integrator(State *state_);

    //! Calculate single point force
    /*!
     * \param computeVirials Virials are computed if this parameter is True
     *
     * Calculate single point energy, i.e. calculate energy only for \link Fix
     * Fixes \endlink with Fix::forceSingle == True.
     */
    void forceSingle(bool computeVirials);

    //! Write output for all \link WriteConfig WriteConfigs \endlink
    void writeOutput();
};

#endif
