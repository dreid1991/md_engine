#pragma once
#ifndef FIXE3B_GMX_H
#define FIXE3B_GMX_H

#include "globalDefs.h"
#include "Fix.h"
#include "GPUArrayGlobal.h"

#include "ThreeBodyE3B_GMX.h" // includes EvaluatorE3B
#include "GridGPU.h"
#include "Molecule.h"
#include "GPUData.h"

//! Make FixE3B available to the python interface
void export_FixE3B_GMX();

//! Explicit 3-Body Potential, v3 (E3B) for Water
/*
 * This fix implements the E3B water for water as 
 * described by Tainter, Shi, & Skinner in 
 * J. Chem. Theory Comput. 2015, 11, 2268-2277
 *
 * This is a ported over version of the CPU code
 *
 * Note that this fix should only be used in conjunction 
 * with water modeled as TIP4P/2005
 */

class FixE3B_GMX: public Fix {
    private: 

        // far cutoff, rf = 5.2 Angstroms; constant, export as readonly
        double rf;

        // short cutoff, rs = 5.0 Angstroms; contsant, export as readonly
        double rs;

        // cutoff of the neighborlist, rf + 1 Angstroms; constant, export as readonly
        double rc;

        // implicitly defined by rc - rf = padding = 1.0 Angstroms; constant, export as readonly
        double padding;

        
        //!< List of int4 atom ids for the list of molecules;
        //   The order of this list does /not/ change throughout the simulation
        GPUArrayDeviceGlobal<int4> waterIdsGPU;
        
        std::vector<int4> waterIds;
 
        // the evaluator for E3B
        EvaluatorE3B_GMX evaluator;

        //!< List of int4 atom idxs for the list of molecules of idxs
        //   The order of this list, and the atom idxs within a given item, change every time \textit{either}: 
        //            (1) state->gpd.idToIdxs changes
        //            (2) gpdLocal.idToIdxs changes
        GPUArrayDeviceGlobal<int4> waterIdxsGPU;

        /* Stuff copied from the Gromacs implementation.. */
        // this will be horribly inefficient on the GPU
        int size; // waterIds.size() * 4 * 10 (4 atoms per molecule, then multiply by 10 (?))
        GPUArrayDeviceGlobal<real4> forces_b2a1;    // as molecules neighborlist size
        GPUArrayDeviceGlobal<real4> forces_c2a1;
        GPUArrayDeviceGlobal<real4> forces_b1a2;
        GPUArrayDeviceGlobal<real4> forces_c1a2;
        GPUArrayDeviceGlobal<real4> pairPairEnergies; // as molecules neighborlist
        GPUArrayDeviceGlobal<real4> pairPairTotal;    // as molecules
        GPUArrayDeviceGlobal<uint>   computeThis;       // size of nlist, set to zero after each computation
        


    public:
        /* FixE3B constructor
         * -- pointer to state
         * -- handle for the fix
         * -- style (E3B2, E3B3)
         *
         *  In the constructor, we set the cutoffs required by this potential.
         */
        FixE3B_GMX(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string style);

        //!< List of all water molecules in simulation
        std::vector<Molecule> waterMolecules;
       
        int nMolecules; // waterMolecules.size();
        
        // corresponding local GPU data; note that we only really need xs - no need for fs, vs, etc..
        // NOTE: this is where we shuffle data by idx; our waterIdsGPU array above remains constant.
        GPUData gpdLocal;
        
        // the local gridGPU for E3B, where we make our molecule by molecule neighborlist
        GridGPU gridGPULocal;
        //! Prepare Fix
        /*!
         * \returns Always returns True
         *
         * This function needs to be called before simulation run.
         */
        bool prepareForRun() override;

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

        // calls our map
        void handleLocalData() override;

        std::string style; // either E3B3 or E3B2
        
        // creates the evaluator for the corresponding style (E3B3, or E3B2);
 
        void createEvaluator();


        void takeStateNThreadPerBlock(int);
        void takeStateNThreadPerAtom(int);
};



#endif /* FIXE3B_H */
