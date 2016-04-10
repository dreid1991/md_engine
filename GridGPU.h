#pragma once
#ifndef GRID_GPU
#define GRID_GPU

#include <map>
#include <set>

#include "GPUArray.h"
#include "GPUArrayDeviceGlobal.h"

class State;

#define EXCL_MASK (~(3<<30));
//okay, this is going to contain all the kernels needed to do gridding
//can also have it contain the 3d grid for neighbor int2 s

/*! \class GridGPU
 * \brief Simulation grid on the GPU
 *
 * This class defines a simulation grid on the GPU. Typically, the GridGPU will
 * be created by AtomGrid::makeGPU().
 */
class GridGPU {
    bool is2d; //!< True for 2d simulations, else false

    /*! \brief Initialize arrays */
    void initArrays();

    /*! \brief Initialize strem
     *
     * This function currently does nothing.
     *
     * \todo Create stream
     */
    void initStream();

    /*! \brief Verfiy consistency of neightbor list
     *
     * \param neighCut Cutoff distance for neighbor building
     *
     * \return True if neighbor list is built correctly. Else, return False.
     *
     * This function is helpful for debugging purposes, checking that the
     * neighbor listing works as expected.
     */
    bool verifyNeighborlists(float neighCut);
    bool streamCreated; //!< True if a stream was created in initStream()

    /*! \brief Verify that sorting atoms into grid works as expected
     *
     * \param gridIdx Index of the grid in GPUArrayDevicePair
     * \param gridIdxs List of gridLo and gridHi values
     * \param grid Currently unused
     *
     * \return True if sorting is correct. Else returns false.
     *
     * This function is helpful for debugging purposes, checking that the
     * atoms are sorted correctly into the grid cells.
     */
    bool checkSorting(int gridIdx, int *gridIdxs, GPUArrayDeviceGlobal<int> &grid);
    public: 
        GPUArray<uint32_t> perCellArray; //!< Number of atoms in a given grid cell, later starting index of cell in neighborlist
        GPUArray<uint32_t> perBlockArray; //!< Number of neighbors in a GPU block
        GPUArray<uint16_t> perAtomArray; //!< For each atom, store the place in the
                                    //!< grid
        GPUArrayDeviceGlobal<float4> xsLastBuild; //!< Contains the atom positions at
                                            //!< the time of the last build.
        GPUArray<int> buildFlag; //!< If buildFlag[0] == true, neighbor list
                                 //!< will be rebuilt
        float3 ds; //!< Grid spacing in x-, y-, and z-dimension
        float3 dsOrig; //!< Grid spacing at the time of construction
        float3 os; //!< Point of origin (lower value for all bounds)
        int3 ns; //!< Number of grid points in each dimension
        GPUArrayDeviceGlobal<uint> neighborlist; //!< Neighbor list
        State *state; //!< Pointer to the simulation state
        float neighCutoffMax; //!< largest cutoff radius of any interacting pair + padding, default value for grid building

        /*! \brief Constructor
         *
         * \param state_ Pointer to the simulation state
         * \param dx Attempted x-resolution of the simulation grid
         * \param dy Attempted y-resolution of the simulation grid
         * \param dz Attempted z-resolution of the simulation grid
         *
         * Constructor to create Grid with approximate resolution. The final
         * resolution will be the next larger value such that the box size is
         * a multiple of the resolution.
         */
        GridGPU(State *state_, float dx, float dy, float dz);

        /*! \brief Constructor
         *
         * \param state_ Pointer to the simulation state
         * \param ds_ float3 containing the grid resolution
         * \param dsOrig_ float3 of the original grid resolution
         * \param os_ Specifying the point of origin
         * \param ns_ Number of grid points in each dimension
         *
         * This constructor assumes that you know exactly what you are doing.
         * For example, it will not check that the grid resolution is
         * commensurate with the box dimensions.
         *
         * \todo Here it would be nicer to have a constructor that takes the
         *       AtomGrid directly like
         *       explicit GridGPU(AtomGrid const &atomGrid)
         */
        GridGPU(State *state_, float3 ds_, float3 dsOrig_, float3 os_,
                                                        int3 ns_, float maxRCut_);

        /*! \brief Default constructor
         *
         * The default constructor. Does not set any values.
         * \todo Do we need default constructor? It doesn't really make sense
         *       to create a grid without setting at least the number of grid
         *       points.
         */
        GridGPU();

        /*! \brief Destructor
         *
         * Destroys the stream object
         *
         * \todo Streams are not used yet, but the Rule of 5 says that if we
         *       have a destructor, we should also declare copy constructor,
         *       move constructor, copy assignment, and move assignment.
         */
        ~GridGPU();

        /*! \brief Take care of bonded-atoms exclusions
         *
         * For bonded interactions, the positions of the next one (bond), two
         * (angle) or three (dihedral) atoms need to be known. This function
         * takes care that they are included in the grid cell.
         */
        void handleExclusions();

        /*! \brief Remap atoms around periodic boundary conditions
         *
         * \param neighCut Cutoff distance for neighbor interactions.
         * Defaults to max values, stored in class
         * \param doSort Sort the perAtomArray
         * \param forceBuild Force rebuilding of neighbor list
         *
         * This function remaps particles that have moved across a periodic
         * boundary and rebuilds the neighbor list if necessary.
         */
        void periodicBoundaryConditions(float neighCut=-1, bool doSort=true,
                                                        bool forceBuild=false);

        /*! \typedef ExclusionList
         * \brief List of atoms bonded along a chain up to a given depth
         *
         * ExclusionList is a map connecting the atom id with a vector of sets
         * of other atom ids. The map should always be ordered.
         */
		typedef std::map<int, std::vector<std::set<int>>> ExclusionList;

        /*! \brief Check if atoms are more closely connected than a given depth
         *
         * \param exclude ExclusionList containing information about how
         *                closely atoms are already connected.
         * \param atomid Atom ID for the first atom
         * \param otherid Atom ID for the second atom
         * \param depthi One larger than the maximum depth to check
         *
         * \return True if there is already a shorter connections between the
         *          two atoms. Else, return False.
         *
         * This function checks whether two atoms are more closely connected
         * than depth. Atoms are connected if they are connected by a chain of
         * bonds, where the depth is defined as the number of bonds in the
         * shortest chain connecting the atoms. This function assumes that
         * exclusion already contains all connections shorter than depth.
         */
		bool closerThan(const ExclusionList &exclude,
						int atomid, int otherid, int16_t depthi);

        /*! \brief Generate the exclusion list
         *
         * \param maxDepth Build exclusion list up to this depth
         *
         * \return Generated exclusion list
         *
         * Build a list of atoms connected via bonds. The depth of the
         * connection is defined as the minimum number of bonds separating the
         * atoms.
         */
		ExclusionList generateExclusionList(const int16_t maxDepth);
      //  ExclusionList exclusionList;
        GPUArrayDeviceGlobal<int> exclusionIndexes; //!< List of exclusion indices
        GPUArrayDeviceGlobal<uint> exclusionIds; //!< List of excluded atom IDs
        int maxExclusionsPerAtom; //!< Maximum number of exclusions for a
                                  //!< single atom
        int numChecksSinceLastBuild; //!< Number of calls to
                                     //!< periodicBoundaryConditions without
                                     //!< rebuilding neighbor list since last
                                     //!< rebuild.
        cudaStream_t rebuildCheckStream; //!< Cuda Stream for asynchronous data
                                         //!< transfer.

        /*! \brief Copy atom positions to xsLastBuild
         *
         * Copies data from state->gpd.xs to xsLastBuild.
         * \todo Stream is not used so copying is not really asynchronous,
         *       right?
         */
        void copyPositionsAsync();
};
#endif
