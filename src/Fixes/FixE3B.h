#pragma once
#ifndef FIXE3B_H
#define FIXE3B_H

#include "globalDefs.h"
#include "Fix.h"
#include "GPUArrayGlobal.h"

#include "ThreeBodyE3B.h" // includes EvaluatorE3B
#include "GridGPU.h"
#include "Molecule.h"
#include "GPUData.h"

//! Make FixE3B available to the python interface
void export_FixE3B();

//! Explicit 3-Body Potential, v3 (E3B) for Water
/*
 * This fix implements the E3B water for water as 
 * described by Tainter, Shi, & Skinner in 
 * J. Chem. Theory Comput. 2015, 11, 2268-2277
 *
 * Note that this fix should only be used in conjunction 
 * with water modeled as TIP4P/2005
 */

class FixE3B: public Fix {
    public:
        /* FixE3B constructor
         * -- pointer to state
         * -- handle for the fix
         * -- group handle
         *
         *  In the constructor, we set the cutoffs required by this potential.
         */
        FixE3B(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle);
        
        // far cutoff, rf = 5.2 Angstroms; constant, export as readonly
        double rf;

        // short cutoff, rs = 5.0 Angstroms; contsant, export as readonly
        double rs;

        // cutoff of the neighborlist, rf + 1 Angstroms; constant, export as readonly
        double rc;

        // implicitly defined by rc - rf = padding = 1.0 Angstroms; constant, export as readonly
        double padding;

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

        bool stepInit() override;

        void singlePointEng(real *) override;

        void compute(int) override;
        
        //! Reset parameters to before processing
        /*!
        * \param handle String specifying the parameter
        */
        //void handleBoundsChange();
        
        // we actually don't need the M-site for this..
        // but require it anyways, because this shouldonly be used with TIP4P/2005
        // -- takes atom IDs as O, H, H, M (see FixRigid.h, FixRigid.cu)
        void addMolecule(int, int, int, int);

        // TODO : implications of not have setEvalWrapper method? This fix will never offload.
        //!< List of all water molecules in simulation
        std::vector<Molecule> waterMolecules;
       
        int nMolecules; // waterMolecules.size();
        
        // calls our map
        void handleLocalData() override;

        // sets the style
        void setStyle(std::string);

        std::string style; // either E3B3 or E3B2
        // creates the evaluator for the corresponding style (E3B3, or E3B2);
        void createEvaluator();

        //!< List of int4 atom ids for the list of molecules;
        //   The order of this list does /not/ change throughout the simulation
        GPUArrayDeviceGlobal<int4> waterIdsGPU;
        
        std::vector<int4> waterIds;
 
        // the local gridGPU for E3B, where we make our molecule by molecule neighborlist
        GridGPU gridGPULocal;

        // corresponding local GPU data; note that we only really need xs - no need for fs, vs, etc..
        // NOTE: this is where we shuffle data by idx; our waterIdsGPU array above remains constant.
        GPUData gpdLocal;

        // the evaluator for E3B
        EvaluatorE3B evaluator;

        //!< List of int4 atom idxs for the list of molecules of idxs
        //   The order of this list, and the atom idxs within a given item, change every time \textit{either}: 
        //            (1) state->gpd.idToIdxs changes
        //            (2) gpdLocal.idToIdxs changes
        GPUArrayDeviceGlobal<int4> waterIdxsGPU;

        int warpsPerBlock; //!< Number of warps (here, also molecules) to send to a given block for E3B 3-body computation
        int numBlocks;     //!< Number of blocks for E3B threebody kernel; depends on how many molecules per block
        int nThreadsPerMolecule; //!< Number of threads per molecule; declared as an int here for clarity,
        // we send it as the 'nThreadPerAtom' variable to the grid so that we have coherent 
        // traversal of the molecular neighbor list

        void checkNeighborlist();

        void checkTwoBodyCompute();

        // we use; we'll start with 8, and 32 threads per molecule --> 256 threads per block
        int threadsPerBlock; //!< threads per block for E3B threebody kernel
        // TODO: removing a molecule?? -- any method that permits modifying, esp. at runtime, 
        // nAtoms in state gpd would similarly be able to remove atoms (molecules) in this GPD
 

        int oldNThreadPerBlock;
        int oldNThreadPerAtom;
        // -- override Fix's methods for takeStateNThreadPerBlock; we do not want state's parameters 
        //    for this fix
        void takeStateNThreadPerBlock(int) override;
        void takeStateNThreadPerAtom(int)  override;

        // uncomment this and pertinent stuff in .cu file to get a list of maxNumNeighbors;
        // ---- These functions were used in conjunction with a specified density of 1.6 g/mL and 1800 molecule simulation 
        //      to get safe parameters that allow us to skip doing this computation during usual runtime.
        //      5 simulations of 100 ps length were used, in which the neighborlist was reconstructed every turn for E3B.
        //      --- speeds us up by removing a deviceToHost() transfer, and one less kernel
        bool recordMaxNumNeighbors;
        int maxNumNeighbors;
        int oldMaxNumNeighbors;
        bool computeMaxNumNeighborsEveryTurn;
        std::vector<int> listOfMaxNumNeighbors;   // !< Let's check how many neighbors we ever have at a given moment;
        std::vector<int> getMaxNumNeighbors();

};



#endif /* FIXE3B_H */
