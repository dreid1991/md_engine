#pragma once

#include "BoundsGPU.h"
#include "cutils_func.h"
#include "Virial.h"
#include "helpers.h"
#include "SquareVector.h"
#include "cutils_math.h"
#include "EvaluatorE3B_GMX.h"

// ifdef __CUDACC__ required otherwise complains when making host side objects;
#ifdef __CUDACC__
// used in compute_E3B_force_twobody to determine whether we continue to process
// TODO: nPerRingPoly
//  --- otherwise, this is complete
template <bool COMP_VIRIALS, bool MULTITHREADPERATOM> 
__global__ void compute_E3B_force_twobody_2b
        (int nMolecules,
         int nPerRingPoly,
         int4 *atomsFromMolecules,     // atom idxs at a given mol idx
         const real4 *__restrict__ xs, // atom positions (as idx)
         real4 *__restrict__ fs,       // atom forces (as idx)
         const uint16_t *__restrict__ neighborCounts, // molecule nlist
         const uint *__restrict__ neighborlist,       // molecule nlist
         const uint32_t * __restrict__ cumulSumMaxPerBlock, //molecule nlist
         int warpSize,                 // device constant
         real4 *e3bTotals,             // total
         real4 *e3bEnergies,           // es
         real4 *pairPairForces,        // pair pair forces
         uint *computeThis,            // boolean flag (1, or 0) for given pair of molecules
         int nThreadPerAtom,
         BoundsGPU bounds,
         Virial *__restrict__ virials, 
         EvaluatorE3B_GMX eval
        )
{
    
    // if multi-thread per atom, then we need to account for nlist traversal
    int idx = GETIDX();
    int molIdx;
    real3 fs_a1a2;
    Virial virial_a1a2;
    real3 totalThisMol; 
    
    extern __shared__ real3 smem_fs[];
    real3 *smem_total;
    Virial *smem_virials;
    if (MULTITHREADPERATOM) {
        smem_virials = (Virial *) (smem_fs + blockDim.x);
        smem_total      = (real3 * ) (smem_virials + blockDim.x);
    }

    // account for NTPA or else you will have a bad time..
    if (idx < nMolecules * nThreadPerAtom) {
       
        if (MULTITHREADPERATOM) {
            molIdx = idx / nThreadPerAtom;
        } else {
            molIdx = idx;
        }
        int4 atomIdxs = atomsFromMolecules[molIdx]; // atom idxs O, H, H, M

        real4 pos_a1_whole = xs[atomIdxs.x];
        real4 pos_b1_whole = xs[atomIdxs.y];
        real4 pos_c1_whole = xs[atomIdxs.z];

        real3 pos_a1 = make_real3(pos_a1_whole);
        real3 pos_b1 = make_real3(pos_b1_whole);
        real3 pos_c1 = make_real3(pos_c1_whole);
        
        fs_a1a2 = make_real3(0.0, 0.0, 0.0);
        totalThisMol = make_real3(0.0, 0.0, 0.0);
        if (COMP_VIRIALS) {
            virial_a1a2 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        
        // load where this molecule's neighbor list starts...
        int baseIdx;
        int memThisBlock;
        int warpsPerBlock = blockDim.x / warpSize;
        int ringPolyIdx = molIdx / nPerRingPoly;	// which ring polymer
        //int beadIdx     = molIdx % nPerRingPoly;	// which time slice // TODO something with this
        // --- see PairEvaluateIso for what to do with beadIdx...
        if (MULTITHREADPERATOM) {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx, nThreadPerAtom);
            int nAtomPerBlock = blockDim.x / nThreadPerAtom;
            int      blockIdx           = ringPolyIdx / nAtomPerBlock;
            uint32_t cumulSumUpToMe     = cumulSumMaxPerBlock[blockIdx];
            uint32_t memSizePerWarpMe   = cumulSumMaxPerBlock[blockIdx+1] - cumulSumUpToMe;
            memThisBlock = warpsPerBlock * (memSizePerWarpMe - cumulSumUpToMe);

        } else {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx);
            int      blockIdx           = ringPolyIdx / blockDim.x;
            uint32_t cumulSumUpToMe     = cumulSumMaxPerBlock[blockIdx];
            uint32_t memSizePerWarpMe   = cumulSumMaxPerBlock[blockIdx+1] - cumulSumUpToMe;
            memThisBlock = warpsPerBlock * (memSizePerWarpMe - cumulSumUpToMe);
        }
       
        int baseIdxE3B = 4 * baseIdx;
        
        // every time we do 'handleLocalData' this will be updated (when the nlist is updated in e3b.
        // --- the 'total', 'es', and fs arrays should be zeroed at the beginning of a given turn
        
        int myIdxInTeam;
        if (MULTITHREADPERATOM) {
            myIdxInTeam = threadIdx.x % nThreadPerAtom; // 0..... nThreadPerAtom - 1
        } else {
            myIdxInTeam = 0;
        }

        // need to change for PIMD
        int numNeigh = neighborCounts[molIdx];

        // loop over the neighborlist...
        for (int nthNeigh=myIdxInTeam; nthNeigh<numNeigh; nthNeigh+=nThreadPerAtom) {
            int nlistIdx;
            if (MULTITHREADPERATOM) {
                nlistIdx = baseIdx + myIdxInTeam + warpSize * (nthNeigh/nThreadPerAtom);
            } else {
                nlistIdx = baseIdx + warpSize * nthNeigh;
            }

            int otherMolIdx = neighborlist[nlistIdx];
            
            // ok, now get other atoms from this molecule
            int4 otherAtomIdxs = atomsFromMolecules[otherMolIdx];

            // get positions
            real4 pos_a2_whole = xs[otherAtomIdxs.x];
            real3 pos_a2       = make_real3(pos_a2_whole);

            // compute O-O dr vector, corresponding force.
            real3 dr_a1a2      = bounds.minImage(pos_a1 - pos_a2);
            real r_a1a2        = length(dr_a1a2);
            if (COMP_VIRIALS) {
                eval.twoBodyForce<true>(dr_a1a2,fs_a1a2,virial_a1a2);
            } else {
                eval.twoBodyForce<false>(dr_a1a2,fs_a1a2,virial_a1a2);
            }
            
        } // nlist loop
        if (MULTITHREADPERATOM) {
            smem_fs[threadIdx.x] = fs_a1a2;
            reduceByN_NOSYNC<real3>(smem_fs, nThreadPerAtom);
            if (myIdxInTeam==0) {
                real4 forceCur = fs[atomIdxs.x]; 
                forceCur += smem_fs[threadIdx.x];
                fs[atomIdxs.x] = forceCur;
            }
            if (COMP_VIRIALS) {
                smem_virials[threadIdx.x] = virial_a1a2;
                reduceByN_NOSYNC<Virial>(smem_virials, nThreadPerAtom);
                // NOTE: we want to multiply this by 0.5, because the two-body 
                //       part of it was computed with the full neighborlist, 
                //       which does not utilize Newton's law.
                if (myIdxInTeam==0) {
                    Virial tmp = smem_virials[threadIdx.x] * 0.5;
                    virials[atomIdxs.x] += tmp;
                }
            }

        } else {
            real4 forceCur = fs[atomIdxs.x]; 
            forceCur += fs_a1a2;
            fs[atomIdxs.x] = forceCur;
            /*
            real4  curTotal = e3bTotals[molIdx];
            curTotal += make_real4(totalThisMol);
            e3bTotals[molIdx] = curTotal;
            */
            if (COMP_VIRIALS) {
                // NOTE: we want to multiply this by 0.5, because the two-body 
                //       part of it was computed with the full neighborlist, 
                //       which does not utilize Newton's law.
                virial_a1a2 *= 0.5;
                virials[atomIdxs.x] += virial_a1a2;
            }
        }
    } // molIdx < nMolecules;
}


// compute the twobody energy for e3b - and some prep for threebody energy routines
template <bool MULTITHREADPERATOM>
__global__ void compute_E3B_energy_twobody_2b
        (int nMolecules,
         int nPerRingPoly,
         int4 *atomsFromMolecules,     // atom idxs at a given mol idx
         const real4 *__restrict__ xs, // atom positions (as idx)
         const uint16_t *__restrict__ neighborCounts, // molecule nlist
         const uint *__restrict__ neighborlist,       // molecule nlist
         const uint32_t * __restrict__ cumulSumMaxPerBlock, //molecule nlist
         int warpSize,                 // device constant
         BoundsGPU bounds,
         real4 *e3bTotals,             // total
         real4 *e3bEnergies,           // es
         uint *computeThis,          // boolean flag (1, or 0) for given pair of molecules
         real *perParticleEng,
         int nThreadPerAtom,
         EvaluatorE3B_GMX eval
        )
{
   
    // if multi-thread per atom, then we need to account for nlist traversal
    int idx = GETIDX();
    int molIdx;
    real3 totalThisMol; 
    // used to aggregate oxygen-oxygen forces (only ones written to global fs[atomIdxs.x] memory)
    // my shared memory for this block will be as atom positions, by reference molecule... then jMoleculeIdxs, by reference molecule
    extern __shared__ real3 smem[];
    real3 *smem_total = smem;
    real *smem_oo;
    if (MULTITHREADPERATOM) {
        smem_oo = (real *) (smem + blockDim.x);
    }

    // something will need to be done for PIMD
    if (idx < nMolecules * nThreadPerAtom) {
        if (MULTITHREADPERATOM) {
            molIdx = idx / nThreadPerAtom;
        } else {
            molIdx = idx;
        }

        int4 atomIdxs = atomsFromMolecules[molIdx]; // atom idxs O, H, H, M

        real4 pos_a1_whole = xs[atomIdxs.x];
        real4 pos_b1_whole = xs[atomIdxs.y];
        real4 pos_c1_whole = xs[atomIdxs.z];

        real3 pos_a1 = make_real3(pos_a1_whole);
        real3 pos_b1 = make_real3(pos_b1_whole);
        real3 pos_c1 = make_real3(pos_c1_whole);
        
        totalThisMol = make_real3(0.0, 0.0, 0.0);
        
        real es_a1 = 0.0;
        // load where this molecule's neighbor list starts...
        int baseIdx;
        int memThisBlock;
        int warpsPerBlock = blockDim.x / warpSize;
        int ringPolyIdx = molIdx/ nPerRingPoly;	// which ring polymer
        //int beadIdx     = molIdx % nPerRingPoly;	// which time slice
        if (MULTITHREADPERATOM) {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx, nThreadPerAtom);
            int nAtomPerBlock = blockDim.x / nThreadPerAtom;
            int      blockIdx           = ringPolyIdx / nAtomPerBlock;
            uint32_t cumulSumUpToMe     = cumulSumMaxPerBlock[blockIdx];
            uint32_t memSizePerWarpMe   = cumulSumMaxPerBlock[blockIdx+1] - cumulSumUpToMe;
            memThisBlock = warpsPerBlock * (memSizePerWarpMe - cumulSumUpToMe);

        } else {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx);
            int      blockIdx           = ringPolyIdx / blockDim.x;
            uint32_t cumulSumUpToMe     = cumulSumMaxPerBlock[blockIdx];
            uint32_t memSizePerWarpMe   = cumulSumMaxPerBlock[blockIdx+1] - cumulSumUpToMe;
            memThisBlock = warpsPerBlock * (memSizePerWarpMe - cumulSumUpToMe);
        }
       
        int baseIdxE3B = 4 * baseIdx;
        
        // every time we do 'handleLocalData' this will be updated (when the nlist is updated in e3b.
        // --- the 'total', 'es', and fs arrays should be zeroed at the beginning of a given turn
        
        int myIdxInTeam;
        if (MULTITHREADPERATOM) {
            myIdxInTeam = threadIdx.x % nThreadPerAtom; // 0..... nThreadPerAtom - 1
        } else {
            myIdxInTeam = 0;
        }

        // need to change for PIMD
        int numNeigh = neighborCounts[molIdx];

        // loop over the neighborlist...
        for (int nthNeigh=myIdxInTeam; nthNeigh<numNeigh; nthNeigh+=nThreadPerAtom) {

            int nlistIdx;
            //int pairPairIdx;
            if (MULTITHREADPERATOM) {
                nlistIdx = baseIdx + myIdxInTeam + warpSize * (nthNeigh/nThreadPerAtom);
            } else {
                nlistIdx = baseIdx + warpSize * nthNeigh;
            }

            int otherMolIdx = neighborlist[nlistIdx];
            
            // ok, now get other atoms from this molecule
            int4 otherAtomIdxs = atomsFromMolecules[otherMolIdx];

            // get positions
            real4 pos_a2_whole = xs[otherAtomIdxs.x];
            real3 pos_a2       = make_real3(pos_a2_whole);

            // compute O-O dr vector, corresponding force.
            real3 dr_a1a2      = bounds.minImage(pos_a1 - pos_a2);

            real r_a1a2 = length(dr_a1a2);
            //  have this return 0.5 * energy, so we can avoid doing atomics for this calculation
            es_a1     += eval.twoBodyCorrectionEnergy(r_a1a2);
        } // nlist loop
        if (MULTITHREADPERATOM) {
            smem_oo[threadIdx.x] = es_a1;

            reduceByN_NOSYNC<real>(smem_oo, nThreadPerAtom);

            if (myIdxInTeam==0) {
                // because other threads wrote to this
                real ppe = perParticleEng[atomIdxs.x];
                ppe += smem_oo[threadIdx.x];
                perParticleEng[atomIdxs.x] = ppe;
            }

        } else {
            real ppe = perParticleEng[atomIdxs.x];
            ppe += es_a1;
            perParticleEng[atomIdxs.x] = ppe;
        }
    } // molIdx < nMolecules;
} 

#endif /* __CUDACC__ */


