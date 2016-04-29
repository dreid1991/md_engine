#pragma once
#ifndef ATOMGRID_H
#define ATOMGRID_H
#include "Python.h"
#include "Grid.h"
#include "Bounds.h"
#include "Atom.h"
#include <math.h>
#include "OffsetObj.h"
#include <iostream>
#include <assert.h>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "globalDefs.h"
#include "Bond.h"
#include "boost_for_export.h"

class State;

using namespace std;
using namespace boost;

void export_AtomGrid();

//! Store Atom pointers on the grid
/*!
 * This class stores Atom pointers on a Grid. The data stored in this class can
 * be converted to a GridGPU using AtomGrid::makeGPU() which will then be sent
 * to the GPU.
 *
 * \todo I think neighCut should be a member of the AtomGrid.
 */
class AtomGrid : public Grid<Atom *> {

private:
    //! Does nothing
    /*!
     * \todo Remove this function or let it issue a warning that it doesn't do
     *       anything.
     */
    void populateGrid() {
    }

    //! Append neighbors to Atoms
    /*!
     * \param a Atom to which the neighbors are appended
     * \param gridSqr One square of the grid, containing neighbor candidates
     * \param boundsTrace Trace of the boundaries
     * \param neighCutSqr Square of the neighbor interaction cutoff distance
     *
     * This function appends all Atoms from a given square of the grid
     * (typically the square the atom resides in plus all neighboring squares)
     * and appends these Atoms to the neighbor list if they are closer than the
     * neighbor interaction cutoff distance.
     */
    void appendNeighborList(Atom &a,
                            OffsetObj<Atom **> &gridSqr,
                            Vector boundsTrace,
                            double neighCutSqr);

    //! Append neighbors to Atoms, check for self-neighboring
    /*!
     * \param a Atom to which the neighbors are appended
     * \param gridSqr One square of the Grid containing neighbor candidates
     * \param boundsTrace Trace of the simulation box boundaries
     * \param neighCutSqr Square of the neighbor interaction cutoff distance
     *
     * This function is almost identical to AtomGrid::appendNeighborList(), but
     * additionally checks for self-neighboring.
     *
     * \todo Is this function really necessary? It is called in
     *       AtomGrid::buildNeighborlistRedun() but not in
     *       AtomGrid::buildNeighborlists(). Either always or never check for
     *       self-neighboring.
     */
    void appendNeighborListSelfCheck(Atom &a,
                                     OffsetObj<Atom **> &gridSqr,
                                     Vector boundsTrace,
                                     double neighCutSqr);

    //! Reset and fill vector containing vector of neighbouring squares
    void setNeighborSquares();

    //! Wrap Atoms around periodic boundary conditions for unskewed box
    /*!
     * \param bounds Bounds to be used for wrapping.
     *
     * This function wraps Atoms around periodic boundary conditions assuming
     * an unskewed simulation box. Typically, the box is unskewed, then Atoms
     * are wrapped and then the box is skewed again.
     *
     * \todo bounds should always be state->bounds or am I wrong?
     */
    void enforcePeriodicUnskewed(Bounds bounds);

    State *state; //!< Pointer to the simulation state

    //! Initialize the AtomGrid
    /*!
     * \param dx_ Attempted resolution in x dimension
     * \param dy_ Attempted resolution in y dimension
     * \param dz_ Attempted resolution in z dimension
     *
     * Initialize the Grid. The total grid dimension must be a multiple of dx,
     * dy, and dz. If not, the resolution is increased to the next larger
     * commensurate value. All Grid elements are filled with NULL pointers.
     */
    void init(double dx_, double dy_, double dz_);

    Bounds boundsOnGridding; //!< Boundaries when the Grid was created
public:
    //! List containing neighboring squares for each square
    /*!
     * This vector contains a vector for each square of the Grid. The vector
     * contained stores a list of OffsetObj specifying the neighboring Grid
     * cells. The OffsetObj contains the state->bounds.trace as the offset and
     * a pointer to the Atom pointers stored in the respective neighbor grid.
     *
     * \todo Why do we need the Offset here if it is just state->bounds.trace?
     */
    vector<vector<OffsetObj<Atom **> > > neighborSquaress;

    double angX; //!< Unused
    double angY; //!< Unused

    //! Unused function - not implemented
    /*!
     * \param vec Unused
     * \todo Remove this function
     */
    void updateAtoms(vector<Atom *> &vec);

    bool isSet; // True if the Grid has been initialized

    //! Create a GridGPU from the AtomGrid
    /*!
     * \param maxRCut Maximum cutoff distance to be set for the GridGPU
     * \return GridGPU Grid to be stored on the GPU
     */
    GridGPU makeGPU(float maxRCut);

    //! Default Constructor
    AtomGrid() : isSet(false) {};

    //! Constructor
    /*!
     * \param state_ Pointer to the simulation state
     * \param dx_ Attempted resolution in x dimension
     * \param dy_ Attempted resolution in y dimension
     * \param dz_ Attempted resolution in z dimension
     */
    AtomGrid(State *state_, double dx_, double dy_, double dz_);

    //! Constructor
    /*!
     * \param state_ Shared pointer to the simulation state
     * \param dx_ Attempted resolution in x dimension
     * \param dy_ Attempted resolution in y dimension
     * \param dz_ Attempted resolution in z dimension
     */
    AtomGrid(SHARED(State) state_, double dx_, double dy_, double dz_);

    //! Wrap atoms around periodic boundaries
    /*!
     * \param bounds Boundaries to wrap atoms around
     *
     * \todo Shouldn't bounds always be state->bounds?
     */
    void enforcePeriodic(Bounds bounds);

    //! Adjust the grid if simulation box changed
    /*!
     * \return True if resizing was successful, False if bounds shrank
     *
     * Adjust the Grid for changed simulation box. Note that gridding may fail
     * if the simulation box has shrunk in at least one dimension. In this case,
     * the function returns False.
     */
    bool adjustForChangedBounds();

    //! Wrap atoms around periodic boundaries
    /*!
     * Calls AtomGrid::periodicBoundaryConditions(double) with the default
     * neighbor interaction distance.
     */
    void periodicBoundaryConditions();

    //! Wrap atoms around periodic boundaries
    /*!
     * \param neighCut Neighbor interaction cutoff
     *
     * Wrap Atoms around periodic boundaries.
     *
     * \todo This function is very close or identical to enforcePeriodic.
     * \todo What happens if boundaries are not periodic?
     */
    void periodicBoundaryConditions(double neighCut);

    //! Build the neighbor lists for all Atoms
    /*!
     * \param neighCut Cutoff distance for neighbor interactions
     *
     * Build neighbor lists for all atoms.
     */
    void buildNeighborlists(double neighCut);

    //! Build neighbor list for only one Atom
    /*!
     * \param a Pointer to Atom whose neighbor list will be built
     * \param neighCut Neighbor cutoff distance
     */
    void buildNeighborlistRedund(Atom *a, double neighCut);

    //! Resize the grid to match state bounds
    /*!
     * \param scaleAtomCoords Change Atom positions such that relative position
     *                        in the box is unchanged.
     *
     * Resize the grid to match the simulation box boundaries in State class
     * again.
     *
     * Note, that it has not been tested yet if this method works for skewed
     * simulation boxes.
     */
    void resizeToStateBounds(bool scaleAtomCoords);

    //! Delete neighbor lists for all Atoms
    void deleteNeighbors();

    //! Build linked lists of Atoms in one Grid square
    /*!
     * Each Grid square stores a pointer to one Atom in the square. The Atom
     * then stores a Pointer to the next Atom in the same square and so on. In
     * other words, Atoms are a linked list.
     *
     * This function builds the linked lists of Atoms.
     */
    void populateLists();
};


#endif
