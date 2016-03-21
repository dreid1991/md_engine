#ifndef ATOM_PARAMS_H
#define ATOM_PARAMS_H

#include "Python.h"
#include <math.h>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <assert.h>

//herm, would like to allow for different force fields
//
//
//id maps to index of data in mass, sigmas, epsilons
//
//this class does not hold per-atom info
#include "boost_for_export.h"
void export_AtomParams();

using namespace boost;
using namespace std;

class State;

/*! \class AtomParams
 * \brief Class storing all available general info on atoms
 *
 * This class stores and manages all available, general info on atoms, such as
 * the number of atom types, their masses and Number in the periodic table.
 */
class AtomParams {
public:

    State *state; //!< Vector to the corresponding state class

    /*! \brief Default constuctor */
    AtomParams() : numTypes(0) {};

    /*! \brief constructor
     *
     * \param s Pointer to the corresponding state
     *
     * Constructor setting the pointer to the corresponding state.
     */
    AtomParams(State *s) : state(s), numTypes(0) {};

    vector<string> handles; //!< List of handles to specify atom types
    vector<double> masses; //!< List of masses, one for each atom type
    vector<double> atomicNums; //!< For each atom type, this vector stores
                               //!< its number in the periodic table

    /*! \brief Return atom type for a given handle
     *
     * \param handle Unique identifier for the atom type
     * \returns Integer specifying the atom type
     *
     * Get the atom type for a given handle.
     */
    int typeFromHandle(string handle);
    int numTypes; //!< Number of atom types

    /*! \brief Add a new type of atoms to the system
     *
     * \param handle Unique identifier for the atoms
     * \param mass Mass of the atoms
     * \param atomicNum Position in the periodic table
     * \returns -1 if handle already exists and updated number of atom
     *          types otherwise.
     *
     * Add a new type of atoms to the system.
     */
    int addSpecies(string handle, double mass, double atomicNum=6);

    /*! \brief Remove all atom type info
     *
     * Delete all info on atom types previously stored in this class.
     */
    void clear();
};

#endif
