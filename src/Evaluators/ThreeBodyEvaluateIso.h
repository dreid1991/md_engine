#pragma once
#ifndef THREE_BODY_EVALUATE_ISO.H
#define THREE_BODY_EVALUATE_ISO.H

#include "BoundsGPU.h"
#include "cutils_func.h"
#include "Virial.h"
#include "helpers.h"
#include "SquareVector.h"

// consider: what needs to be templated here? //
// -- we need number of molecules, a neighborlist /by molecule/
// ---- this neighborlist should be a list of integer molecule ids,
//      for which there is a map organized by molecule id from which we 
//      can extract the atom ids (and from there our usual pos, force, vel, etc.)
__global__ void compute_three_body_iso 
        (int nMolecules, 
         const float4 *__restrict__ xs, 
         float4 *__restrict__ fs, 
         const uint16_t *__restrict__ neighborCounts, 
         const uint *__restrict__ neighborlist, 
         const uint32_t * __restrict__ cumulSumMaxPerBlock, 
         int warpSize, 
         BoundsGPU bounds, 
         Virial *__restrict__ virials) 
{

    __syncthreads();
    
    int idx = GETIDX();

    if (idx < nMolecules) {

        // stuff with virials here as well

        // so, we'll need to make baseMoleculeNeighList().. for now, just call it as if we have it
        // -- the purpose of this is to load the neighbors associated with this molecule ID
        int baseMoleculeIdx = baseMoleculeNeighList(cumulSumMaxPerBlock, warpSize);

        // number of neighbors this molecule has, with which it can form trimers
        int numNeighMolecules = neighborCountsMolecule[idx];
       
        // here we should extract the positions of the O, H atoms of this water molecule
        // first, get the atom indices - maybe this will be stored as an array of ints?

        // is this the correct way to access this...? assuming it will be of similar form to neighborlist,
        // except now we need a list of atoms corresponding to a given molecule..
        // additionally, we assume that the list of atoms in a molecule is ordered as {O, H1, H2}
        uint* atomsMolecule1 = atomsFromMolecule[idx];
    
        /* NOTE to others: see the notation used in 
         * Kumar and Skinner, J. Phys. Chem. B., 2008, 112, 8311-8318
         * "Water Simulation Model with Explicit 3 Body Interactions"
         *
         * we use their notation for decomposing the molecules into constituent atoms a,b,c (oxygen, hydrogen, hydrogen)
         * and decomposing the given trimer into the set of molecules 1,2,3 (water molecule 1, 2, and 3)
         */
       
        // copy the float4 vectors of the positions
        float4 pos_a1_whole = xs[iAtoms[0]];
        float4 pos_b1_whole = xs[iAtoms[1]];
        float4 pos_c1_whole = xs[iAtoms[2]];

        // now, get just positions in float3
        float3 pos_a1 = make_float3(pos_a1_whole);
        float3 pos_b1 = make_float3(pos_b1_whole);
        float3 pos_c1 = make_float3(pos_c1_whole);

        // extract the initial forces on these atoms; these will be modified at the end of this function
        float4 fs_a1_whole = fs[iAtoms[0]];
        float4 fs_b1_whole = fs[iAtoms[1]];
        float4 fs_c1_whole = fs[iAtoms[2]];

        // create a new force sum variable for these atoms
        float3 fs_a1_sum = make_float3(0.0, 0.0, 0.0);
        float3 fs_b1_sum = make_float3(0.0, 0.0, 0.0);
        float3 fs_c1_sum = make_float3(0.0, 0.0, 0.0);
        
        // iterate over the neighbors of molecule '1'; compute the two-body interaction w.r.t all it's neighbors
        // when they are denoted as molecule '2' (or, correspondingly, j)
        for (int j = 0; j < (numNeighMolecules); j++) {
            
            // get idx of this molecule
            // this does assume that molecules are grouped according to warpsize
            // -- then, the atomIDs that we need are somehow accessible via MoleculeID
            int jlistMoleculeIdx = baseMoleculeIdx + warpSize * j;
            int jrawIdx = neighborlist[jlistMoleculeIdx];
            
            uint* atomsMolecule2 = atomsFromMolecule[jrawIdx];
            float4 pos_a2_whole = xs[jAtoms[0]];
            float4 pos_b2_whole = xs[jAtoms[1]];
            float4 pos_c2_whole = xs[jAtoms[2]];
    
            // here we should extract the positions for the O, H atoms of this water molecule
            float3 pos_a2 = make_float3(pos_a2_whole);
            float3 pos_b2 = make_float3(pos_b2_whole);
            float3 pos_c2 = make_float3(pos_c2_whole);

            // we have four OH distances to compute here
            float3 r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
            float3 r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
           
            float3 r_a2b1 = bounds.minImage(pos_a2 - pos_b1);
            float3 r_a2c1 = bounds.minImage(pos_a2 - pos_c2);

            // we now have our molecule 'j'
            // compute the two-body correction term w.r.t the oxygens
            float3 r_a1a2 = bounds.minImage(pos_a1 - pos_a2);
            float r_a1a2_magnitude = length(r_a1a2);

            fs_a1_sum += eval.twoBodyForce(r_a1a2,r_a1a2_magnitude)
            // -- verify that our molecule 'j' is actually within the bounds of our potential; else, continue
            bool moleculePairInBounds = evaluator.checkBounds(r_a1b2,r_a1c2,r_a2b1,r_a2c1);
            

            // if this pair of molecules has /no/ intermolecular O-H distance less than the max cutoff,
            // all three-body terms will be zero.
            if (!moleculePairInBounds) continue;

            // --> get molecule 'k' to complete the trimer

            // we only wish to compute $-/nabla E_{ijk}$ for all unique combos of trimers, so this should range 
            // from k = j+1, while still less than numNeighMolecules w.r.t. baseMolecule ('i');
            // --- we only need to continue with this iteration if at least /one/ of the hydrogens above was within 
            //     the three-body far cutoff
            //     --- this is accomplished by boolean 'moleculePairInBounds'
            
            for (int k = j+1; k < numNeighMolecules; k++) {
                // grab warp index corresponding to this 'k'
                int klistMoleculeIdx = baseMoleculeIdx + warpSize * k;
                // convert this index to a molecule index within our molecule array
                int krawIdx = neighborlist[klistMoleculeIdx];

                // we now have our k molecule
                uint* atomsMolecule3 = atomsFromMolecule[krawIdx];

                // extract positions of O, H atoms of this water molecule
                float4 pos_a3_whole = xs[kAtoms[0]];
                float4 pos_b3_whole = xs[kAtoms[1]];
                float4 pos_c3_whole = xs[kAtoms[2]];

                float3 pos_a3 = make_float3(pos_a3_whole);
                float3 pos_b3 = make_float3(pos_b3_whole);
                float3 pos_c3 = make_float3(pos_c3_whole);
                
                // compute the pertinent O-H distances for use in our potential (there are 8 that we have yet to compute)
                
                // -- distances vector for a1b3 and a1c3:
                float3 r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
                float3 r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
               
                // -- distances vector for a2b3 and a2c3:
                float3 r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
                float3 r_a2c3 = bounds.minImage(pos_a2 - pos_c3);

                // -- distances vector for a3b1 and a3c1:
                float3 r_a3b1 = bounds.minImage(pos_a3 - pos_b1);
                float3 r_a3c1 = bounds.minImage(pos_a3 - pos_c1);

                // -- distance vector for a3b2 and a3c2:
                float3 r_a3b2 = bounds.minImage(pos_a3 - pos_b2);
                float3 r_a3c2 = bounds.minImage(pos_a3 - pos_c2);

                /*
                 *  get the magnitude of the new distance vectors, and check if we still need to compute this potential
                 *  (i.e., see if this is a valid trimer, that there will be some non-zero threebody contribution)
                 */










            }
        }
    }
}


// also, need to compute the potential energy (in addition to the forces)

#endif





