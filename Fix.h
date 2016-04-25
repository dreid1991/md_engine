#pragma once
#ifndef FIX_H
#define FIX_H

#include "Python.h"
#include "globalDefs.h"
#include "Atom.h"
#include "list_macro.h"
#include <iostream>
#include "GPUArrayGlobal.h"
#include "GPUArrayTex.h"
#include "State.h"
#include "FixTypes.h"
#include <pugixml.hpp>


#include "boost_for_export.h"
void export_Fix();
//#include "DataManager.h"
using namespace std;

//! Base class for Fixes
/*!
 * Fixes modify the dynamics in the system. They are called by the Integrater
 * at regular intervals and they can modify the Atom forces, positions and
 * velocities. Note that as Fixes themselves depend on the current forces,
 * positions and velocities, the order in which the Fixes are defined and
 * called is important.
 *
 * For this reason, Fixes have an order preference. Fixes with a low preference
 * are computed first.
 *
 * \todo Compile list of preferences in FixTypes.h. I think each Fix should
 *       have an order preference.
 */
class Fix {
public:
    State *state; //!< Pointer to the simulation state
    string handle; //!< "Name" of the Fix
    string groupHandle; //!< Group to which the Fix applies
    int applyEvery; //!< Applyt this fix every this many timesteps
    unsigned int groupTag; //!< Bitmask for the group handle
    string type; //!< Unused. \todo Check if really unused, then remove

    //! Default constructor
    /*!
     * \todo Make constructor protected since this is an abstract base class
     */
    Fix() {}

    //! Constructor
    /*!
     * \param state_ Pointer to simulation state
     * \param handle_ Name of the Fix
     * \param groupHandle_ String specifying on which group the Fix acts
     * \param type_ Type of the Fix (unused?)
     * \param applyEvery_ Apply Fix every this many timesteps
     *
     * \todo Make constructor protected since this is an abstract base class
     */
    Fix(SHARED(State) state_,
        string handle_,
        string groupHandle_,
        string type_,
        int applyEvery_);

    //! Apply fix
    /*!
     * \param computeVirials Compute virials for this Fix
     *
     * \todo Make purely virtual.
     */
    virtual void compute(bool computeVirials) {}

    //! Calculate single point energy of this Fix
    /*!
     * \param perParticleEng Pointer to where to store the per-particle energy
     *
     * The pointer passed needs to be a pointer to a memory location on the
     * GPU.
     *
     * \todo Use cuPointerGetAttribute() to check that the pointer passed is a
     *       pointer to GPU memory.
     *
     * \todo Make this function purely virtual.
     */
    virtual void singlePointEng(float *perParticleEng) {}

    //! Perform calculations at the end of a simulation run
    /*!
     * Some Fixes set up internal variables in the Fix::prepareForRun()
     * function. This function then typically sets these values back to their
     * default.
     *
     * \todo Make this function purely virtual
     */
    virtual void postRun() {}

    //! Test if another Fix is the same
    /*!
     * \param f Reference of Fix to test
     * \return True if they have the same handle
     *
     * Two Fixes are considered equal if they have the same "name" stored in
     * handle. Thus, a FixLJCut called "myFix" and a FixSpringStatic called
     * "myFix" are considered equal.
     *
     * \todo Take const reference, make function const
     * \todo Why not use operator==()?
     * \todo Isn't comparing the handle overly simplistic?
     */
    bool isEqual(Fix &f);

    bool forceSingle; //!< True if this Fix contributes to single point energy.
    int orderPreference; //!< Fixes with a high order preference are calculated
                         //!< later.

    //! Recalculate group bitmask from a (possibly changed) handle
    void updateGroupTag();

    //! Accomodate for new type of Atoms added to the system
    /*!
     * \param handle String specifying the new type of Atoms
     *
     * \todo Make purely virtual.
     */
    virtual void addSpecies(string handle) {}
    virtual bool prepareForRun() {return true;};
    virtual bool downloadFromRun() {return true;};
    virtual ~Fix() {};
    virtual bool refreshAtoms(){return true;};
    virtual bool readFromRestart(pugi::xml_node restData){return true;};
    virtual string restartChunk(string format){return "";};
    //virtual vector<pair<int, vector<int> > > neighborlistExclusions();
    const string restartHandle;
    // TODO: think about treatment of different kinds of bonds in fixes
    // right now for ease, each vector of bonds in any given fix that stores
    // bonds has to store them in a vector<BondVariant> variable
    // you can push_back, insert, whatever, other kinds of bonds into this
    // vector
    // you have to get them out using a getBond method, or using the
    // boost::get<BondType>(vec) syntax
    // it's not perfect, but it lets us generically collect vectors without
    // doing any copying
    virtual vector<BondVariant> *getBonds() {
        return nullptr;
    }
    virtual vector<float> getRCuts() {
        return vector<float>();
    }
    void validAtoms(vector<Atom *> &atoms);
};

/*
do it with precompiler instructions, lol!
nah, just do methods of state.  Might have to add other function calls later as fixes become more complicated
*/
#endif
