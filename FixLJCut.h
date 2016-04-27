#pragma once
#ifndef FIXLJCUT_H
#define FIXLJCUT_H

#include "FixPair.h"
#include "xml_func.h"

//! Make FixLJCut available to the pair base class in boost
void export_FixLJCut();

/*! \class FixLJCut
 * \brief Fix for truncated Lennard-Jones interactions
 *
 * Fix to calculate Lennard-Jones interactions of particles. Note that the
 * cutoff distance rcut is defined in the class State.
 */
#include "EvaluatorLJ.h"
class FixLJCut : public FixPair {
    public:
        /*! \brief Constructor */
        FixLJCut(SHARED(State), string handle);

        //! Compute forces
        void compute(bool);
        //! Compute single point energy
        void singlePointEng(float *);
        /*! \brief Prepare Fix
         *
         * \returns Always returns True
         *
         * This function needs to be called before the first run.
         */
        bool prepareForRun();

        /*! \brief Create restart string
         *
         * \param format Format of the pair parameters.
         *
         * \returns restart chunk string.
         */
        string restartChunk(string format);

        /*! \brief Read parameters from restart file
         *
         * \param restData XML node containing the restart data.
         */
        bool readFromRestart(pugi::xml_node restData);

        /*! \brief Add new type of atoms
         *
         * \param handle Not used
         *
         * This function adds a new particle type to the fix.
         */
        void addSpecies(string handle);

        //* Member variables *//
        const string epsHandle; //!< Handle for parameter epsilon
        const string sigHandle; //!< Handle for parameter sigma
        const string rCutHandle; //!< Handle for parameter rCut
        vector<float> epsilons;
        vector<float> sigmas;
        vector<float> rCuts;
        vector<float> getRCuts();

        void postRun();
        EvaluatorLJ evaluator;
};

#endif
