#pragma once

#include "BoundsGPU.h"
#include "cutils_func.h"
#include "Virial.h"
#include "helpers.h"
#include "SquareVector.h"
#include "cutils_math.h"
#include "EvaluatorE3B.h"

// ifdef __CUDACC__ required otherwise complains when making host side objects;
#ifdef __CUDACC__
// used in compute_E3B_force_twobody to determine whether we continue to process
// --- assumes Bernal-Fowler geometry (TIP4P or TIP4P/2005) and a base cutoff of 5.2 Angstroms
// // i.e., 5.2 (\AA) + 0.9572 O-H bond length.
#define RMAX_E3B 6.157200
//  --- otherwise, this is complete
template <bool COMP_VIRIALS, bool MULTITHREADPERATOM> 
__global__ void compute_E3B_force_twobody
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
         real4 *forces_b2a1,
         real4 *forces_c2a1,
         real4 *forces_b1a2,
         real4 *forces_c1a2,
         uint *computeThis,            // boolean flag (1, or 0) for given pair of molecules
         int nThreadPerAtom,
         BoundsGPU bounds,
         Virial *__restrict__ virials, 
         EvaluatorE3B eval
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
        int ringPolyIdx = molIdx/ nPerRingPoly;	// which ring polymer
        // --- see PairEvaluateIso for what to do with beadIdx...
        if (MULTITHREADPERATOM) {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx, nThreadPerAtom);

        } else {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx);
        }
       
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
            // RMAX_E3B is defined as 0.52 + 0.09572 = 0.61572;
            // we do as GMX implementation.
            // --- molIdx < otherMolIdx bc we don't want to overcount things.
            //     this condition is invariant bc we launch the kernels sequentially
            //     i.e. molIdxs are not shuffled between kernel launches
            if (r_a1a2 <= RMAX_E3B and molIdx < otherMolIdx) {
                // get the positions of the H1, H2 associated with otherMolIdx
                real4 pos_b2_whole = xs[otherAtomIdxs.y];
                real4 pos_c2_whole = xs[otherAtomIdxs.z];

                real3 pos_b2 = make_real3(pos_b2_whole);
                real3 pos_c2 = make_real3(pos_c2_whole);

                // dr vectors - do as rH - rO, always
                real3 dr_b2a1 = bounds.minImage(pos_b2 - pos_a1);
                real3 dr_c2a1 = bounds.minImage(pos_c2 - pos_a1);
                real3 dr_b1a2 = bounds.minImage(pos_b1 - pos_a2);
                real3 dr_c1a2 = bounds.minImage(pos_c1 - pos_a2);

                real   r_b2a1 = length(dr_b2a1);
                real   r_c2a1 = length(dr_c2a1);
                real   r_b1a2 = length(dr_b1a2);
                real   r_c1a2 = length(dr_c1a2);

                real es_b2a1 = eval.threeBodyPairEnergy(r_b2a1);
                real es_c2a1 = eval.threeBodyPairEnergy(r_c2a1);
                real es_b1a2 = eval.threeBodyPairEnergy(r_b1a2);
                real es_c1a2 = eval.threeBodyPairEnergy(r_c1a2);

                real3 fs_b2a1 = eval.threeBodyForce(dr_b2a1, r_b2a1);
                real3 fs_c2a1 = eval.threeBodyForce(dr_c2a1, r_c2a1);
                real3 fs_b1a2 = eval.threeBodyForce(dr_b1a2, r_b1a2);
                real3 fs_c1a2 = eval.threeBodyForce(dr_c1a2, r_c1a2);

                // add energies to global atom arrays (e3b data arrays)
                // a1 contributions

                real4 esOtherMol = make_real4(es_b1a2 + es_c1a2, es_b2a1, es_c2a1, 0.0);
                totalThisMol += make_real3(es_b2a1 + es_c2a1, es_b1a2, es_c1a2);
                
                forces_b2a1[nlistIdx] = make_real4(fs_b2a1);
                forces_c2a1[nlistIdx] = make_real4(fs_c2a1);
                forces_b1a2[nlistIdx] = make_real4(fs_b1a2);
                forces_c1a2[nlistIdx] = make_real4(fs_c1a2);

                // same index as in nlist
                e3bEnergies[nlistIdx]    = make_real4(es_b2a1, es_c2a1, es_b1a2, es_c1a2);
                // atomic add bc other threads are possibly computing stuff for molecule 'otherMolIdx'
                // -- need to use preprocessor macros, otherwise we get compiler errors.
#ifdef DASH_DOUBLE
                double * addr = &(e3bTotals[otherMolIdx].x);
#else
                float * addr = &(e3bTotals[otherMolIdx].x);
#endif
                atomicAdd(addr + 0, esOtherMol.x);
                atomicAdd(addr + 1, esOtherMol.y);
                atomicAdd(addr + 2, esOtherMol.z);
                atomicAdd(addr + 3, esOtherMol.w);
                computeThis[nlistIdx] = 1; // else false
            } // e3b inner calculations
       
        } // nlist loop
        if (MULTITHREADPERATOM) {
            smem_fs[threadIdx.x] = fs_a1a2;
            reduceByN_NOSYNC<real3>(smem_fs, nThreadPerAtom);
            smem_total[threadIdx.x] = totalThisMol;
            reduceByN_NOSYNC<real3>(smem_total, nThreadPerAtom);
            if (myIdxInTeam==0) {
                real4 forceCur = fs[atomIdxs.x]; 
                forceCur += smem_fs[threadIdx.x];
                fs[atomIdxs.x] = forceCur;
                
                real3 esSumSmem = smem_total[threadIdx.x];
#ifdef DASH_DOUBLE
                double * addr_es = &(e3bTotals[molIdx].x);
#else
                float * addr_es  = &(e3bTotals[molIdx].x);
#endif
                atomicAdd(addr_es + 0, esSumSmem.x);
                atomicAdd(addr_es + 1, esSumSmem.y);
                atomicAdd(addr_es + 2, esSumSmem.z);
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
#ifdef DASH_DOUBLE
            double * addr_es = &(e3bTotals[molIdx].x);
#else
            float *  addr_es  = &(e3bTotals[molIdx].x);
#endif
            // atomic add bc we don't know when other blocks are writing to e3bTotals[molIdx].
            atomicAdd(addr_es + 0, totalThisMol.x);
            atomicAdd(addr_es + 1, totalThisMol.y);
            atomicAdd(addr_es + 2, totalThisMol.z);
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



// compute_E3B_force_center computes all of the forces for triplets (i,j,k) 
// -- where j, k are both on i's neighborlist
template <bool COMP_VIRIALS, bool MULTITHREADPERATOM> 
// TODO: nPerRingPoly, COMP_VIRIALS, smem limitation.
//       -- should probably compute a few properties before bothering with the smem limitations..
//          Also, could make smem allocation contingent on COMP_VIRIALS,
//          which would make NVT able to be optimized further for runtime
__global__ void compute_E3B_force_threebody
        (int nMolecules,
         int nPerRingPoly,
         int4 *atomsFromMolecules,     // atom idxs at a given mol idx
         const real4 *__restrict__ xs, // atom positions (as idx)
         real4 *fs,       // atom forces (as idx)
         const uint16_t *__restrict__ neighborCounts, // molecule nlist
         const uint *__restrict__ neighborlist,       // molecule nlist
         const uint32_t * __restrict__ cumulSumMaxPerBlock, //molecule nlist
         int warpSize,                 // device constant
         BoundsGPU bounds,
         real4 *e3bTotals,             // total - arranged as molecules
         real4 *e3bEnergies,           // es    - arranged as molecule nlist
         real4 *forces_b2a1,
         real4 *forces_c2a1,
         real4 *forces_b1a2,
         real4 *forces_c1a2,
         uint *computeThis,            // boolean flag (1, or 0) for given pair of molecules
         Virial *virials, 
         int nThreadPerAtom,
         EvaluatorE3B eval
        )
{
    

    real3 fs_a1, fs_b1, fs_c1; // need to aggregate across threads if multithreadperatom
    
    // we'll store fs_a1, fs_b1, fs_c1 for thisMolIdx and then reduce across the block 
    // if there is multi-thread per atom
    extern __shared__ real3 smem_fs[];
    real3* smem_fs_a1 = smem_fs;
    real3* smem_fs_b1;
    real3* smem_fs_c1;
    Virial *smem_virials_a1;
    Virial *smem_virials_b1;
    Virial *smem_virials_c1;
    if (MULTITHREADPERATOM) {
        smem_fs_b1   = smem_fs_a1 + blockDim.x;
        smem_fs_c1   = smem_fs_b1 + blockDim.x;
        smem_virials_a1 = (Virial *) (smem_fs_c1 + blockDim.x);
        smem_virials_b1 = smem_virials_a1 + blockDim.x;
        smem_virials_c1 = smem_virials_b1 + blockDim.x;
    }
    
    // same thing with virials
    Virial virial_a1;
    Virial virial_b1;
    Virial virial_c1;

    // if multi-thread per atom, then we need to account for nlist traversal
    int idx = GETIDX();
    int molIdx;
    if (idx < nMolecules * nThreadPerAtom) {
        if (MULTITHREADPERATOM) {
            molIdx = idx/nThreadPerAtom;
        } else {
            molIdx = idx;
        }
        int4 atomIdxs = atomsFromMolecules[molIdx]; // atom idxs O, H, H, M

        fs_a1 = make_real3(0.0, 0.0, 0.0);
        fs_b1 = make_real3(0.0, 0.0, 0.0);
        fs_c1 = make_real3(0.0, 0.0, 0.0);

        if (COMP_VIRIALS) {
            virial_a1 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            virial_b1 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            virial_c1 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
       

        // load where this molecule's neighbor list starts...
        int baseIdx;
        int ringPolyIdx = molIdx/ nPerRingPoly;	// which ring polymer
        if (MULTITHREADPERATOM) {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx, nThreadPerAtom);
        } else {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx);
        }
        real4 totalsThisMol = e3bTotals[molIdx];
        
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
        //printf("molIdx %d has %d numNeigh!\n", molIdx, numNeigh);
        real3 fs_a2, fs_b2, fs_c2;

        real4 esThisPair;
        real4 fs_b2a1_read, fs_c2a1_read, fs_b1a2_read, fs_c1a2_read;
        real3 fs_b2a1, fs_c2a1, fs_b1a2, fs_c1a2;
        Virial virial_a2, virial_b2, virial_c2;
        // don't need to grab positions of other atom - just fs, energies etc.
        // loop over the neighborlist...
        
        for (int nthNeigh=myIdxInTeam; nthNeigh<numNeigh; nthNeigh+=nThreadPerAtom) {
            fs_a2 = make_real3(0.0, 0.0, 0.0);
            fs_b2 = make_real3(0.0, 0.0, 0.0);
            fs_c2 = make_real3(0.0, 0.0, 0.0);

            int nlistIdx;
            if (MULTITHREADPERATOM) {
                nlistIdx = baseIdx + myIdxInTeam + warpSize * (nthNeigh/nThreadPerAtom);
            } else {
                nlistIdx = baseIdx + warpSize * nthNeigh;
            }
 
            if (COMP_VIRIALS) {
                virial_a2 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                virial_b2 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                virial_c2 = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }

            // pairPairIdx provides the indices for forces and energies stored in global arrays;
            // totals are accessible by molIdx, yielding a single read for one molecule
            if (computeThis[nlistIdx]) {

                // ok, we have pairPairIdx_ of the associated interactions
                // --  note that thisMolIdx < otherMolIdx by virtue of computeThis[nlistIdx] only 
                //     having been tripped in the previous, complementary kernel;
                int otherMolIdx = neighborlist[nlistIdx];
                
                // ok, now get other atoms from this molecule - for writing to forces
                int4 otherAtomIdxs = atomsFromMolecules[otherMolIdx];
                
                // need the total associated with this molecule, as well as energies of pairPair interactions
                real4 totalsOtherMol = e3bTotals[otherMolIdx];
                esThisPair = e3bEnergies[nlistIdx];

                // des with itmp = ntmp*12 + (k-1)*3, k == 1, 2
                //
                fs_b2a1_read = forces_b2a1[nlistIdx];
                fs_b2a1 = make_real3(fs_b2a1_read);
                
                fs_c2a1_read = forces_c2a1[nlistIdx];
                fs_c2a1 = make_real3(fs_c2a1_read);

                // des with itmp = ntmp*12 + (k+1)*3, k == 1, 2
                fs_b1a2_read = forces_b1a2[nlistIdx];
                fs_b1a2 = make_real3(fs_b1a2_read);
                
                fs_c1a2_read = forces_c1a2[nlistIdx];
                fs_c1a2 = make_real3(fs_c1a2_read);

                // write to fs_a1, fs_b1, fs_c1, fs_a2, fs_b2, fs_c2 etc. others are just reads
                // TODO this is where we put in the COMP_VIRIALS stuff.
                //  also, make that a template - if we aren't doing NPT, then we just want it to not be there.
                if (COMP_VIRIALS) {
                    eval.threeBodyPairForce<true>(fs_b2a1, fs_c2a1, fs_b1a2, fs_c1a2,
                                            totalsThisMol, totalsOtherMol, esThisPair,
                                            fs_a1, fs_b1, fs_c1,
                                            fs_a2, fs_b2, fs_c2);
                } else { 
                    eval.threeBodyPairForce<false>(fs_b2a1, fs_c2a1, fs_b1a2, fs_c1a2,
                                            totalsThisMol, totalsOtherMol, esThisPair,
                                            fs_a1, fs_b1, fs_c1,
                                            fs_a2, fs_b2, fs_c2);
                }

                // atomic writes to otherMolIdxs' atoms.
#ifdef DASH_DOUBLE
                double * addr_fs_o = &((fs[otherAtomIdxs.x]).x);
                double * addr_fs_h1= &((fs[otherAtomIdxs.y]).x);
                double * addr_fs_h2= &((fs[otherAtomIdxs.z]).x);
#else
                float * addr_fs_o = &((fs[otherAtomIdxs.x]).x);
                float * addr_fs_h1= &((fs[otherAtomIdxs.y]).x);
                float * addr_fs_h2= &((fs[otherAtomIdxs.z]).x);
#endif
                atomicAdd(addr_fs_o+0 ,fs_a2.x);
                atomicAdd(addr_fs_o+1 ,fs_a2.y);
                atomicAdd(addr_fs_o+2 ,fs_a2.z);

                atomicAdd(addr_fs_h1+0,fs_b2.x);
                atomicAdd(addr_fs_h1+1,fs_b2.y);
                atomicAdd(addr_fs_h1+2,fs_b2.z);

                atomicAdd(addr_fs_h2+0,fs_c2.x);
                atomicAdd(addr_fs_h2+1,fs_c2.y);
                atomicAdd(addr_fs_h2+2,fs_c2.z);

                // atomic writes to otherMolIdxs' atoms' virials
                if (COMP_VIRIALS) {
                    // do something here, with the things we computed before
                }
                
                
                // something else??
                // need to do a global write after this for otherMolIdx, with atomic operations.
                // can just aggregate thisMol without doing atomics and assign at the conclusion of 
                // the kernel.

            } // end computeThis
        } //  end nlist iteration

        if (MULTITHREADPERATOM) {
            smem_fs_a1[threadIdx.x] = fs_a1;
            smem_fs_b1[threadIdx.x] = fs_b1;
            smem_fs_c1[threadIdx.x] = fs_c1;
            reduceByN_NOSYNC<real3>(smem_fs_a1, nThreadPerAtom);
            reduceByN_NOSYNC<real3>(smem_fs_b1, nThreadPerAtom);
            reduceByN_NOSYNC<real3>(smem_fs_c1, nThreadPerAtom);
            if (myIdxInTeam==0) {

                real3 fs_o_aggregate = smem_fs_a1[threadIdx.x];
                real3 fs_h1_aggregate= smem_fs_b1[threadIdx.x];
                real3 fs_h2_aggregate= smem_fs_c1[threadIdx.x];
#ifdef DASH_DOUBLE
                double * addr_fs_o = &((fs[atomIdxs.x]).x);
                double * addr_fs_h1= &((fs[atomIdxs.y]).x);
                double * addr_fs_h2= &((fs[atomIdxs.z]).x);
                
#else
                float * addr_fs_o = &((fs[atomIdxs.x]).x);
                float * addr_fs_h1= &((fs[atomIdxs.y]).x);
                float * addr_fs_h2= &((fs[atomIdxs.z]).x);

#endif
                atomicAdd(addr_fs_o+0,fs_o_aggregate.x);
                atomicAdd(addr_fs_o+1,fs_o_aggregate.y);
                atomicAdd(addr_fs_o+2,fs_o_aggregate.z);

                atomicAdd(addr_fs_h1+0,fs_h1_aggregate.x);
                atomicAdd(addr_fs_h1+1,fs_h1_aggregate.y);
                atomicAdd(addr_fs_h1+2,fs_h1_aggregate.z);

                atomicAdd(addr_fs_h2+0,fs_h2_aggregate.x);
                atomicAdd(addr_fs_h2+1,fs_h2_aggregate.y);
                atomicAdd(addr_fs_h2+2,fs_h2_aggregate.z);
            }
            // TODO
            if (COMP_VIRIALS) {
                // assign..
                smem_virials_a1[threadIdx.x] = virial_a1;
                smem_virials_b1[threadIdx.x] = virial_b1;
                smem_virials_c1[threadIdx.x] = virial_c1;
                // reduce..
                reduceByN_NOSYNC<Virial>(smem_virials_a1, nThreadPerAtom);
                reduceByN_NOSYNC<Virial>(smem_virials_b1, nThreadPerAtom);
                reduceByN_NOSYNC<Virial>(smem_virials_c1, nThreadPerAtom);
                
                // write to global..
                if (myIdxInTeam == 0) {

                // TODO

                }
            }
        } else {

#ifdef DASH_DOUBLE
            double * addr_fs_o = &((fs[atomIdxs.x]).x);
            double * addr_fs_h1= &((fs[atomIdxs.y]).x);
            double * addr_fs_h2= &((fs[atomIdxs.z]).x);
                
#else
            float * addr_fs_o = &((fs[atomIdxs.x]).x);
            float * addr_fs_h1= &((fs[atomIdxs.y]).x);
            float * addr_fs_h2= &((fs[atomIdxs.z]).x);

#endif
            
            atomicAdd(addr_fs_o+0 ,fs_a1.x);
            atomicAdd(addr_fs_o+1 ,fs_a1.y);
            atomicAdd(addr_fs_o+2 ,fs_a1.z);

            atomicAdd(addr_fs_h1+0,fs_b1.x);
            atomicAdd(addr_fs_h1+1,fs_b1.y);
            atomicAdd(addr_fs_h1+2,fs_b1.z);

            atomicAdd(addr_fs_h2+0,fs_c1.x);
            atomicAdd(addr_fs_h2+1,fs_c1.y);
            atomicAdd(addr_fs_h2+2,fs_c1.z);
            
            if (COMP_VIRIALS) {

                // read from global, add, then write. do for a1, b1, c1
                // -- these should not require atomicAdds...? I think.
            }
        }

    } // end if molIdx < nMolecules
} // end kernel;


// compute the twobody energy for e3b - and some prep for threebody energy routines
template <bool MULTITHREADPERATOM>
__global__ void compute_E3B_energy_twobody
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
         EvaluatorE3B eval
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
        int ringPolyIdx = molIdx/ nPerRingPoly;	// which ring polymer
        if (MULTITHREADPERATOM) {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx, nThreadPerAtom);

        } else {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx);
        }
       
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

            // RMAX_E3B is defined as 0.52 + 0.09572 = 0.61572;
            // we do as GMX implementation.
            if (r_a1a2 <= RMAX_E3B and molIdx < otherMolIdx) {
                // get the positions of the H1, H2 associated with otherMolIdx
                real4 pos_b2_whole = xs[otherAtomIdxs.y];
                real4 pos_c2_whole = xs[otherAtomIdxs.z];

                real3 pos_b2 = make_real3(pos_b2_whole);
                real3 pos_c2 = make_real3(pos_c2_whole);

                // dr vectors - do as rH - rO, always
                real3 dr_b2a1 = bounds.minImage(pos_b2 - pos_a1);
                real3 dr_c2a1 = bounds.minImage(pos_c2 - pos_a1);
                real3 dr_b1a2 = bounds.minImage(pos_b1 - pos_a2);
                real3 dr_c1a2 = bounds.minImage(pos_c1 - pos_a2);

                real   r_b2a1 = length(dr_b2a1);
                real   r_c2a1 = length(dr_c2a1);
                real   r_b1a2 = length(dr_b1a2);
                real   r_c1a2 = length(dr_c1a2);

                real es_b2a1 = eval.threeBodyPairEnergy(r_b2a1);
                real es_c2a1 = eval.threeBodyPairEnergy(r_c2a1);
                real es_b1a2 = eval.threeBodyPairEnergy(r_b1a2);
                real es_c1a2 = eval.threeBodyPairEnergy(r_c1a2);

                // add energies to global atom arrays (e3b data arrays)
                // a1 contributions

                real4 esOtherMol = make_real4(es_b1a2 + es_c1a2, es_b2a1, es_c2a1, 0.0);
                totalThisMol += make_real3(es_b2a1 + es_c2a1, es_b1a2, es_c1a2);
                
                // same index as in nlist
                e3bEnergies[nlistIdx]    = make_real4(es_b2a1, es_c2a1, es_b1a2, es_c1a2);
                // atomic add bc other threads are possibly computing stuff for molecule 'otherMolIdx'
#ifdef DASH_DOUBLE
                double * addr = &(e3bTotals[otherMolIdx].x);
#else
                float * addr = &(e3bTotals[otherMolIdx].x);
#endif
                atomicAdd(addr + 0, esOtherMol.x);
                atomicAdd(addr + 1, esOtherMol.y);
                atomicAdd(addr + 2, esOtherMol.z);
                atomicAdd(addr + 3, esOtherMol.w);
                computeThis[nlistIdx] = 1; // else false
            } // e3b inner calculations
       
        } // nlist loop
        if (MULTITHREADPERATOM) {
            smem_oo[threadIdx.x] = es_a1;
            smem_total[threadIdx.x] = totalThisMol;

            reduceByN_NOSYNC<real3>(smem_total, nThreadPerAtom);
            reduceByN_NOSYNC<real>(smem_oo, nThreadPerAtom);

            if (myIdxInTeam==0) {
                // because other threads wrote to this
                //real4 esCur = e3bTotals[molIdx];
                real3 esSumSmem = smem_total[threadIdx.x];
#ifdef DASH_DOUBLE
                double * addr_molIdx = &(e3bTotals[molIdx].x);
#else
                float *  addr_molIdx = &(e3bTotals[molIdx].x);
#endif
                atomicAdd(addr_molIdx + 0, esSumSmem.x);
                atomicAdd(addr_molIdx + 1, esSumSmem.y);
                atomicAdd(addr_molIdx + 2, esSumSmem.z);

                // no atomicAdd needed here; we only write to this atomIdxs.x on this thread.
                real ppe = perParticleEng[atomIdxs.x];
                ppe += smem_oo[threadIdx.x];
                perParticleEng[atomIdxs.x] = ppe;
            }

        } else {
#ifdef DASH_DOUBLE
            double * addr_molIdx = &(e3bTotals[molIdx].x);
#else
            float *  addr_molIdx = &(e3bTotals[molIdx].x);
#endif
            atomicAdd(addr_molIdx + 0, totalThisMol.x);
            atomicAdd(addr_molIdx + 1, totalThisMol.y);
            atomicAdd(addr_molIdx + 2, totalThisMol.z);

            real ppe = perParticleEng[atomIdxs.x];
            ppe += es_a1;
            perParticleEng[atomIdxs.x] = ppe;
        }
    } // molIdx < nMolecules;
} 


// compute e3b energy for triplet i-j-k, j and k on i's neighborlist
template <bool MULTITHREADPERATOM>
__global__ void compute_E3B_energy_threebody
        (int nMolecules,
         int nPerRingPoly,
         int4 *atomsFromMolecules,     // atom idxs at a given mol idx
         const real4 *__restrict__ xs, // atom positions (as idx)
         const uint16_t *__restrict__ neighborCounts, // molecule nlist
         const uint *__restrict__ neighborlist,       // molecule nlist
         const uint32_t * __restrict__ cumulSumMaxPerBlock, //molecule nlist
         int warpSize,                 // device constant
         real4 *e3bTotals,             // total
         real4 *e3bEnergies,           // es
         uint *computeThis,          // boolean flag (1, or 0) for given pair of molecules
         real *perParticleEng,
         int nThreadPerAtom,
         EvaluatorE3B eval
        )
{
    
    // if multi-thread per atom, then we need to account for nlist traversal
    int idx = GETIDX();
    int molIdx;
    // we'll store fs_a1, fs_b1, fs_c1 for thisMolIdx and then reduce across the block 
    // if there is multi-thread per atom
    extern __shared__ real3 smem[];
    real3* smem_es = smem;
    
    // something will need to be done for PIMD
    if (idx < nMolecules * nThreadPerAtom) {
        if (MULTITHREADPERATOM) {
            molIdx = idx / nThreadPerAtom;
        } else {
            molIdx = idx;
        }

        real3 myAtomsEnergies = make_real3(0.0, 0.0, 0.0);
        int4 atomIdxs = atomsFromMolecules[molIdx]; // atom idxs O, H, H, M
        // load where this molecule's neighbor list starts...
        int baseIdx;
        int ringPolyIdx = molIdx/ nPerRingPoly;	// which ring polymer
        if (MULTITHREADPERATOM) {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx, nThreadPerAtom);

        } else {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx);
        }
       
        real4 totalsThisMol = e3bTotals[molIdx];
        
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

        real4 esThisPair;
        
        for (int nthNeigh=myIdxInTeam; nthNeigh<numNeigh; nthNeigh+=nThreadPerAtom) {
            real3 otherAtomsEnergies = make_real3(0.0, 0.0, 0.0);

            int nlistIdx;
            if (MULTITHREADPERATOM) {
                nlistIdx = baseIdx + myIdxInTeam + warpSize * (nthNeigh/nThreadPerAtom);
            } else {
                nlistIdx = baseIdx + warpSize * nthNeigh;
            }

            // pairPairIdx provides the indices for forces and energies stored in global arrays;
            // totals are accessible by molIdx, yielding a single read for one molecule
            if (computeThis[nlistIdx]) {

                // ok, we have pairPairIdx_ of the associated interactions
                // --  note that thisMolIdx < otherMolIdx by virtue of computeThis[nlistIdx] only 
                //     having been tripped in the previous, complementary kernel;
                int otherMolIdx = neighborlist[nlistIdx];
                
                // ok, now get other atoms from this molecule - for writing to forces
                int4 otherAtomIdxs = atomsFromMolecules[otherMolIdx];
                
                // need the total associated with this molecule, as well as energies of pairPair interactions
                real4 totalsOtherMol = e3bTotals[otherMolIdx];
                esThisPair = e3bEnergies[nlistIdx];

                // write to fs_a1, fs_b1, fs_c1, fs_a2, fs_b2, fs_c2 etc. others are just reads
                // TODO this is where we put in the COMP_VIRIALS stuff.
                //  also, make that a template - if we aren't doing NPT, then we just want it to not be there.
                eval.threeBodyPairEnergy(totalsThisMol, totalsOtherMol, esThisPair, myAtomsEnergies, otherAtomsEnergies);

                //real4 otherAtomsAsReal4 = make_real4(otherAtomsEnergies);
                atomicAdd(&(perParticleEng[otherAtomIdxs.x]), otherAtomsEnergies.x);
                atomicAdd(&(perParticleEng[otherAtomIdxs.y]), otherAtomsEnergies.y);
                atomicAdd(&(perParticleEng[otherAtomIdxs.z]), otherAtomsEnergies.z);

            } // end computeThis
        } //  end nlist iteration


        if (MULTITHREADPERATOM) {
            smem_es[threadIdx.x] = myAtomsEnergies;
            reduceByN_NOSYNC<real3>(smem_es, nThreadPerAtom);
            if (myIdxInTeam==0) {
                
#ifdef DASH_DOUBLE
                double * addr_ppe_o = &(perParticleEng[atomIdxs.x]);
                double * addr_ppe_h1 = &(perParticleEng[atomIdxs.y]);
                double * addr_ppe_h2 = &(perParticleEng[atomIdxs.z]);
#else
                float * addr_ppe_o = &(perParticleEng[atomIdxs.x]);
                float * addr_ppe_h1 = &(perParticleEng[atomIdxs.y]);
                float * addr_ppe_h2 = &(perParticleEng[atomIdxs.z]);

#endif

                atomicAdd(addr_ppe_o, smem_es[threadIdx.x].x);
                atomicAdd(addr_ppe_h1,smem_es[threadIdx.x].y);
                atomicAdd(addr_ppe_h2,smem_es[threadIdx.x].z);
                // I wonder if putting atomic additions here would reduce the discrepancy..
                /*
                real curEngO = perParticleEng[atomIdxs.x];
                real curEngH1= perParticleEng[atomIdxs.y];
                real curEngH2= perParticleEng[atomIdxs.z];
    
                curEngO += smem_es[threadIdx.x].x;
                curEngH1+= smem_es[threadIdx.x].y;
                curEngH2+= smem_es[threadIdx.x].z;

                perParticleEng[atomIdxs.x] = curEngO;
                perParticleEng[atomIdxs.y] = curEngH1;
                perParticleEng[atomIdxs.z] = curEngH2;
                */
            }
        } else {
                /*
                real curEngO = perParticleEng[atomIdxs.x];
                real curEngH1= perParticleEng[atomIdxs.y];
                real curEngH2= perParticleEng[atomIdxs.z];
    
                curEngO += myAtomsEnergies.x;
                curEngH1+= myAtomsEnergies.y;
                curEngH2+= myAtomsEnergies.z;

                perParticleEng[atomIdxs.x] = curEngO;
                perParticleEng[atomIdxs.y] = curEngH1;
                perParticleEng[atomIdxs.z] = curEngH2;
                */
#ifdef DASH_DOUBLE
                double * addr_ppe_o = &(perParticleEng[atomIdxs.x]);
                double * addr_ppe_h1 = &(perParticleEng[atomIdxs.y]);
                double * addr_ppe_h2 = &(perParticleEng[atomIdxs.z]);
#else
                float * addr_ppe_o = &(perParticleEng[atomIdxs.x]);
                float * addr_ppe_h1 = &(perParticleEng[atomIdxs.y]);
                float * addr_ppe_h2 = &(perParticleEng[atomIdxs.z]);

#endif
                atomicAdd(addr_ppe_o, myAtomsEnergies.x);
                atomicAdd(addr_ppe_h1,myAtomsEnergies.y);
                atomicAdd(addr_ppe_h2,myAtomsEnergies.z);
            
        } // end if molIdx < nMolecules
    }
} // end kernel;


#endif /* __CUDACC__ */


