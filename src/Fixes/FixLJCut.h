#pragma once
#ifndef FIXLJCUT_H
#define FIXLJCUT_H

#include "FixPair.h"
#include "PairEvaluatorLJ.h"
#include "xml_func.h"

//! Make FixLJCut available to the pair base class in boost
void export_FixLJCut();

//! Fix for truncated Lennard-Jones interactions
/*!
 * Fix to calculate Lennard-Jones interactions of particles. Note that the
 * cutoff distance rcut is defined in the class State.
 */
class FixLJCut : public FixPair {
public:
    //! Constructor
    FixLJCut(SHARED(State), std::string handle);

    //! Compute forces
    void compute(bool);

    //! Compute single point energy
    void singlePointEng(float *);

    //! Prepare Fix
    /*!
     * \returns Always returns True
     *
     * This function needs to be called before simulation run.
     */
    bool prepareForRun();

    //! Run after simulation
    /*!
     * This function needs to be called after simulation run.
     */
    bool postRun();

    //! Create restart string
    /*!
     * \param format Format of the pair parameters.
     *
     * \returns restart chunk string.
     */
    std::string restartChunk(std::string format);

    //! Read parameters from restart file
    /*!
     * \return Always True
     *
     * \param restData XML node containing the restart data.
     */
    bool readFromRestart(pugi::xml_node restData);

    //! Add new type of atoms
    /*!
     * \param handle Not used
     *
     * This function adds a new particle type to the fix.
     */
    void addSpecies(std::string handle);

    //! Return list of cutoff values
    std::vector<float> getRCuts();

public:
    const std::string epsHandle; //!< Handle for parameter epsilon
    const std::string sigHandle; //!< Handle for parameter sigma
    const std::string rCutHandle; //!< Handle for parameter rCut
    std::vector<float> epsilons; //!< vector storing epsilon values
    std::vector<float> sigmas; //!< vector storing sigma values
    std::vector<float> rCuts; //!< vector storing cutoff distance values

    EvaluatorLJ evaluator; //!< Evaluator for generic pair interactions
};

#endif
