#pragma once
#ifndef FIXE3B3_H
#define FIXE3B3_H

/* inherits from Fix.h, has a FixTIP4P member */
#include "Fix.h"
#include "GPUArrayGlobal.h"
#include "PairEvaluatorE3B3.h"
#include "ThreeBodyEvaluateIso.h"
#include "ThreeBodyEvaluatorE3B3.h"

//! Make FixE3B3 available to the python interface
void export_FixE3B3();

//! Explicit 3-Body Potential, v3 (E3B3) for Water
/*
 * This fix implements the E3B3 water for water as 
 * described by Tainter, Shi, & Skinner in 
 * J. Chem. Theory Comput. 2015, 11, 2268-2277
 *
 * Note that this fix should only be used in conjunction 
 * with water modeled as TIP4P-2005
 */

class FixE3B3: public Fix {
    
    private:
    
        GPUArrayDeviceGlobal<int4> waterIdsGPU;
        GPUArrayDeviceGlobal<float4> xs_0;
        GPUArrayDeviceGlobal<float4> vs_0;
        GPUArrayDeviceGlobal<float4> dvs_0;
        GPUArrayDeviceGlobal<float4> fs_0;
    
    
        GPUArrayDeviceGlobal<float4> fix_len;
        std::vector<int4> waterIds;
        std::vector<BondVariant> bonds;
        std::vector<float4> invMassSums;
    
    
    public:

        // delete the default constructor
        FixE3B3() = delete;

        /* FixE3B3 constructor
         * -- pointer to state
         * -- handle for the fix
         * -- group handle
         * -- two-body cutoff 'rcut' - should be the same 
         *    as LJ interactions on regular TIP4P!
         * -- short cutoff for three body 'rs'
         * -- far cutoff for three body 'rf'
         */
        FixE3B3(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle,
                  float rcut, 
                  float rs, 
                  float rf);
        
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

        void compute(bool);
        
        //! Reset parameters to before processing
        /*!
        * \param handle String specifying the parameter
        */
        void handleBoundsChange();
        
        // we actually don't need the M-site for this..
        void createRigid(int, int, int);

        std::vector<BondVariant> *getBonds() {
            return &bonds;
        }
        
        void setEvalWrapper();
        void setEvalWrapperOrig();

        // the evaluators for the forces and energy contributions
        EvaluatorE3B3 twoBodyEvaluator;
        ThreeBodyEvaluatorE3B3 threeBodyEvaluator;

};



#endif /* FIXE3B3_H */
