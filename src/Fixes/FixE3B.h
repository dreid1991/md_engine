#pragma once
#ifndef FIXE3B_H
#define FIXE3B_H

#include "globalDefs.h"
#include "FixPair.h"
#include "Fix.h"
#include "GPUArrayGlobal.h"

/* TODO: move pair interactions in to EvaluatorE3B. */
//#include "PairEvaluatorE3B.h"
#include "EvaluatorE3B.h"
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
    
    private:
   



    public:

        // delete the default constructor
        FixE3B() = delete;

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
        
        // far cutoff, rf = 5.2 Angstroms
        double rf;

        // short cutoff, rs = 5.0 Angstroms
        double rs;

        // cutoff of the neighborlist, rf + 2 Angstroms
        double rc;

        // implicitly defined by rc - rf = padding = 2.0 Angstroms
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

        bool stepInit();

        //void singlePointEng(float *);

        void compute(int);
        
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
        void handleLocalData();

        //!< List of all water molecules in simulation
        std::vector<Molecule> waterMolecules;
       
        int nMolecules; // waterMolecules.size();
        //!< List of int4 atom ids for the list of molecules;
        //   The order of this list does /not/ change throughout the simulation
        GPUArrayDeviceGlobal<int4> waterIdsGPU;
        //!< List of int4 atom idxs for the list of molecules of idxs
        //   The order of this list, and the atom idxs within a given item, change every time \textit{either}: 
        //            (1) state->gpd.idToIdxs changes
        //            (2) gpdLocal.idToIdxs changes
        GPUArrayDeviceGlobal<int4> waterIdxsGPU;
        std::vector<int4> waterIds;
 
        // the local gridGPU for E3B, where we make our molecule by molecule neighborlist
        GridGPU gridGPULocal;

        // corresponding local GPU data; note that we only really need xs - no need for fs, vs, etc..
        GPUData gpdLocal;

        // the evaluator for E3B
        EvaluatorE3B evaluator;

};



#endif /* FIXE3B_H */
