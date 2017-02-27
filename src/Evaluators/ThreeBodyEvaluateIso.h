#pragma once
#ifndef THREE_BODY_EVALUATE_ISO.H
#define THREE_BODY_EVALUATE_ISO.H

#include "BoundsGPU.h"
#include "cutils_func.h"
#include "Virial.h"
#include "helpers.h"
#include "SquareVector.h"

// consider: what needs to be templated here? //
// TODO plenty of work to be done;
// -- we need number of molecules, a neighborlist /by molecule/
// ---- this neighborlist should be a list of integer molecule ids,
//      for which there is a map organized by molecule id from which we 
//      can extract the atom ids (and from there our usual pos, force, vel, etc.)
//
//
// really, for now, don't worry about making this 'iso' so much as making it work for E3B3
__global__ void compute_three_body_iso 
        (int nMolecules, 
         const float4 *__restrict__ xs, 
         float4 *__restrict__ fs, 
         const uint16_t *__restrict__ neighborCounts, 
         const uint *__restrict__ neighborlist, 
         const uint32_t * __restrict__ cumulSumMaxPerBlock, 
         int warpSize, 
         const float *__restrict__ parameters, 
         int numTypes,  
         BoundsGPU bounds, 
         float onetwoStr, 
         float onethreeStr, 
         float onefourStr, 
         Virial *__restrict__ virials, 
         float *qs, 
         float qCutoffSqr, 
         PAIR_EVAL pairEval, 
         CHARGE_EVAL chargeEval) 
{

    __syncthreads();
    int idx = GETIDX();

    if (idx < nMoleculess) {

        // stuff with virials here as well

        // so, we'll need to make baseMoleculeNeighList().. for now, just call it as if we have it
        // -- 
        int baseMoleculeIdx = baseMoleculeNeighList(cumulSumMaxPerBlock, warpSize);

        // number of neighbors this molecule has, with which it can form trimers
        int numNeighMolecules = neighborCountsMolecule[idx];
       
        // here we should extract the positions of the O, H atoms of this water molecule
        // first, get the atom indices - maybe this will be stored as an array of ints?

        // is this the correct way to access this...? assuming it will be of similar form to neighborlist,
        // except now we need a list of atoms corresponding to a given molecule..
        uint* iAtoms = atomsFromMolecule[idx];
        float4 imol_OPosWhole = xs[iAtoms[0]];
        float4 imol_H1PosWhole = xs[iAtoms[1]];
        float4 imol_H2PosWhole = xs[iAtoms[2]];

        // now, get just positions in float3

        float3 imol_OPos = make_float3(imol_OPosWhole);
        float3 imol_H1Pos = make_float3(imol_H1PosWhole);
        float3 imol_H2Pos = make_float3(imol_H2PosWhole);

        float4 imol_O_fsWhole = fs[iAtoms[0]];
        float4 imol_H1_fsWhole = fs[iAtoms[1]];
        float4 imol_H2_fsWhole = fs[iAtoms[2]];

        // can i separate the sums of into expressions that contribute to just oxygen, just hydrogen1, just hydrogen2...?
        // -- I think not.  Need Mathematica to be sure.
        
        // TODO see if we need this..
        float3 O_forceSum = make_float3(0.0, 0.0, 0.0);
        // we need to range over the entire list of numNeighMolecules, because we need to compute two things:
        // -- the pair correction term E_{ij}
        // -- the trimer contribution
        for (int j = 0; j < (numNeighMolecules); j++) {
            // get idx of this molecule
            // this does assume that molecules are grouped according to warpsize
            // -- then, the atomIDs that we need are somehow accessible via MoleculeID
            int jlistMoleculeIdx = baseMoleculeIdx + warpSize * j;
            int jrawIdx = neighborlist[jlistMoleculeIdx];
            
            uint* jAtoms = atomsFromMolecule[jrawIdx];
            float4 jmol_OPosWhole = xs[jAtoms[0]];
            float4 jmol_H1PosWhole = xs[jAtoms[1]];
            float4 jmol_H2PosWhole = xs[jAtoms[2]];
    
            // here we should extract the positions fo the O, H atoms of this water molecule
            float3 jmol_OPos = make_float3(jmol_OPosWhole);
            float3 jmol_H1Pos = make_float3(jmol_H1PosWhole);
            float3 jmol_H2Pos = make_float3(jmol_H2PosWhole);

            // compute the pertinent O-H distances for use in our potential
            // ----> just for {ij}, compute the O-O distance as well, because we need it for our two-body correction term
            //
            float3 r_OiOj = bounds.minImage(imol_OPos - jmol_OPos);
           
            // we have four OH distances to compute here
            float3 r_OiH1j = bounds.minImage(imol_OPos - jmol_H1Pos);
            float3 r_OiH2j = bounds.minImage(imol_OPos - jmol_H2Pos);
            
            float3 r_OjH1i = bounds.minImage(jmol_OPos - imol_H1Pos);
            float3 r_OjH2i = bounds.minImage(jmol_OPos - jmol_H2Pos);


            float3 correction_term_ij = eval.forceTwoBody(r_OiOj);

            // we now have our molecule 'j'
            // --> get molecule 'k' to complete the trimer

            // we only wish to compute \Delta E_{ijk} for all unique combos of trimers, so this should range 
            // from k = j+1, while still less than numNeighMolecules w.r.t. baseMolecule ('i')
            for (int k = j+1; k < numNeighMolecules; k++) {
                int klistMoleculeIdx = baseMoleculeIdx + warpSize * k;
                int krawIdx = neighborlist[klistMoleculeIdx];

                // we now have our k molecule
                uint* kAtoms = atomsFromMolecule[krawIdx];

                // extract positions of O, H atoms of this water molecule
                float4 kmol_OPosWhole = xs[kAtoms[0]];
                float4 kmol_H1PosWhole = xs[kAtoms[1]];
                float4 kmol_H2PosWhole = xs[kAtoms[2]];

                // compute the pertinent O-H distances for use in our potential (there are 8)
                
                // - distances from 







                // NOTE: for each molecule /baseMolecule/ we do these same loops;
                // so, the force as computed should be directed only on molecule i.
                // additionally, I think this force is directed only w.r.t the atoms it is between?
                // think further on this matter

            }
        }
    }
}


// also, need to compute the potential energy (in addition to the forces)

#endif





