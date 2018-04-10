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

template <bool COMP_VIRIALS> 
__global__ void compute_E3B_force_twobody
        (int nMolecules, 
         const int4 *__restrict__ atomsFromMolecule,  // accessed by molecule idx, as atomidxs
         const uint16_t *__restrict__ neighborCounts, // by moleculeIdx
         const uint *__restrict__ neighborlist,       // bymoleculeIdxs
         const uint32_t * __restrict__ cumulSumMaxPerBlock, // gridGPULocal
         int warpSize, // device property
         const real4 *__restrict__ xs,  // as atom idxs
         real4 *__restrict__ fs,        // as atom idxs
         BoundsGPU bounds, 
         Virial *__restrict__ virials,  // as atom idxs
         int nMoleculesPerBlock,        
         int maxNumNeighbors,
         EvaluatorE3B eval)
{

    // populate smem_neighborIdxs for this warp
    // blockDim.x gives number of threads in a block in x direction
    // gridDim.x gives number of blocks in a grid in the x direction
    // blockDim.x * gridDim.x gives n threads in grid in x direction
    // blockIdx.x * (nMoleculesPerBlock) : e.g., blockIdx.x == 0: 0... then threadIdx.x/warpSize: 0...7
    //        so first 8 molecules
    // blockIdx.x == 1 --> moleculeIdx = [8,..,15], etc. OK.
    // ---- blockIdx.x * (nMoleculesPerBlock) + (threadIdx.x /32) gives moleculeIdx
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // nThreadPerAtom ~ warpSize; this gives our starting point in nlist.
    int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx, warpSize);

    /* declare the variables */
    Virial virialsSum_a;
    Virial agg_virialsSum_a;
    real3 pair_fs_sum_a;
    real3 agg_pair_fs_sum_a;
    int4 atomsReferenceMolecule;
    real3 pos_a1;
    int neighborlistSize, initNlistIdx;
    /* we did that because we assign values and do read-writes in separate if statements, 
     * in between __syncwarp() calls */
    // this sum is only for molecule 1; therefore, drop 1 subscript; a,b,c denote O, H1, H2, respectively
    pair_fs_sum_a = make_real3(0.0, 0.0, 0.0);

    // aggregate virials and forces for the O, H, H of the reference molecules within this warp
    if (COMP_VIRIALS) {
        virialsSum_a = Virial(0,0,0,0,0,0);
    }

    // this will be true or false for an entire warp
    if (moleculeIdx < nMolecules) {
        atomsReferenceMolecule = atomsFromMolecule[moleculeIdx]; // get our reference molecule atom idxs
        // load referencePos of atoms from global memory;
        real4 pos_a1_whole = xs[atomsReferenceMolecule.x];
        // get positions as real3
        pos_a1 = make_real3(pos_a1_whole);
        
        neighborlistSize = neighborCounts[moleculeIdx];
        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        initNlistIdx= threadIdx.x % warpSize; // begins as 0...31
        int curNlistIdx = initNlistIdx;
        real3 xs_O;
        real4 xs_O_whole;
        real3 rij_a1a2;
        // curNlistIdx goes as the set {0...31} + (32 * N), N = 0...ceil(numNeighbors/32)
        
        
        //for (int nthNeigh=myIdxInTeam; nthNeigh<numNeigh; nthNeigh+=nThreadPerAtom) {
        //    int nlistIdx;
        //    if (MULTITHREADPERATOM) {
        //        nlistIdx = baseIdx + myIdxInTeam + warpSize * (nthNeigh/nThreadPerAtom);
        while (curNlistIdx < neighborlistSize) {
            // retrieve the neighborlist molecule idx corresponding to this thread
            int nlistIdx = baseIdx + initNlistIdx +  warpSize * (curNlistIdx/warpSize);
            int neighborMoleculeIdx = neighborlist[nlistIdx];                  // global memory access
            int4 neighborAtomIdxs    = atomsFromMolecule[neighborMoleculeIdx]; // global memory access

            xs_O_whole = xs[neighborAtomIdxs.x]; // global memory access
            
            xs_O  = make_real3(xs_O_whole);
            rij_a1a2 = bounds.minImage(pos_a1 - xs_O);
            
            if (COMP_VIRIALS) {
                eval.twoBodyForce<true>(rij_a1a2,pair_fs_sum_a,virialsSum_a);
            } else {
                eval.twoBodyForce<false>(rij_a1a2,pair_fs_sum_a,virialsSum_a);
            }
            curNlistIdx += warpSize;                                        // advance as a warpSize
        }
     
        // finally, add myself to the end of the list (we check this list later when deciding on a 
    } // end (moleculeIdx < nMolecules)
    __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                  // --- this is cuda 9.0 function
                  //
 

    // we need to wait until all threads are done computing their forces (and virials)
    // now, do lane shifting to accumulate the forces (and virials, if necessary) in to threadIdx == 0 which does global write
    // warpReduce all forces; if virials, warpReduce all Virials as well
    // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
    agg_pair_fs_sum_a = warpReduceSum(pair_fs_sum_a,warpSize);
    if (COMP_VIRIALS) {
        // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
        // virials for oxygen of reference molecule
        agg_virialsSum_a = warpReduceSum(virialsSum_a,warpSize);
    
    }
    // no syncing required after warp reductions
    
    // threadIdx.x % warpSize does global write, iff. moleculeIdx < nMolecules
    if (((threadIdx.x % warpSize) == 0) and (moleculeIdx < nMolecules)) {
        // load curForce on O molecule
        real4 curForce_O = fs[atomsReferenceMolecule.x];

        // add contributions from E3B
        curForce_O += agg_pair_fs_sum_a;

        if (COMP_VIRIALS) {
            // load from global memory
            Virial virial_O = virials[atomsReferenceMolecule.x];
            // add contributions from E3B to global value
            virial_O += agg_virialsSum_a;
            // write to global memory
            virials[atomsReferenceMolecule.x] = virial_O;
        }
        // write forces to global
        fs[atomsReferenceMolecule.x] = curForce_O;
    }

} // end kernel
// compute_E3B_force_center computes all of the forces for triplets (i,j,k) 
// -- where j, k are both on i's neighborlist
template <bool COMP_VIRIALS> 
__global__ void compute_E3B_force_center
        (int nMolecules, 
         const int4 *__restrict__ atomsFromMolecule,  // accessed by molecule idx, as atomidxs
         const uint16_t *__restrict__ neighborCounts, // by moleculeIdx
         const uint *__restrict__ neighborlist,       // bymoleculeIdxs
         const uint32_t * __restrict__ cumulSumMaxPerBlock, // gridGPULocal
         int warpSize, // device property
         const real4 *__restrict__ xs,  // as atom idxs
         real4 *__restrict__ fs,        // as atom idxs
         BoundsGPU bounds, 
         Virial *__restrict__ virials,  // as atom idxs
         int nMoleculesPerBlock,        
         int maxNumNeighbors,
         EvaluatorE3B eval)
{

    // we have one molecule per warp, presumably;
    // store neighborlist as atomIdxsInMolecule
    extern __shared__ real3 smem_neighborAtomPos[];

    real3 *smem_pos = smem_neighborAtomPos;
    // my shared memory for this block will be as atom positions, by reference molecule... then jMoleculeIdxs, by reference molecule
    uint32_t *jMoleculeIdxs = (uint32_t*)&smem_pos[3*maxNumNeighbors*nMoleculesPerBlock]; 

    // populate smem_neighborIdxs for this warp
    // blockDim.x gives number of threads in a block in x direction
    // gridDim.x gives number of blocks in a grid in the x direction
    // blockDim.x * gridDim.x gives n threads in grid in x direction
    // blockIdx.x * (nMoleculesPerBlock) : e.g., blockIdx.x == 0: 0... then threadIdx.x/warpSize: 0...7
    //        so first 8 molecules
    // blockIdx.x == 1 --> moleculeIdx = [8,..,15], etc. OK.
    // ---- blockIdx.x * (nMoleculesPerBlock) + (threadIdx.x /32) gives moleculeIdx
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // get where our neighborlist for this molecule starts // TODO verify
    //int baseIdx = baseNeighlistIdxFromIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx);
    // baseIdx is as if we pass cumulSumMaxPerBlock array, warpSize... then atomIdx is moleculeIdx, 
    // nThreadPerAtom ~ warpSize; this gives our starting point in nlist.
    int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx, warpSize);

    /* declare the variables */
    Virial virialsSum_a;
    Virial virialsSum_b;
    Virial virialsSum_c;
    real3 fs_sum_a, fs_sum_b, fs_sum_c;

    Virial agg_virialsSum_a;
    Virial agg_virialsSum_b;
    Virial agg_virialsSum_c;
    real3 agg_fs_sum_a, agg_fs_sum_b, agg_fs_sum_c;

    int4 atomsReferenceMolecule;
    real3 pos_a1,pos_b1,pos_c1;
    int neighborlistSize, base_smem_idx,initNlistIdx,base_smem_idx_idxs;
    //int maxNumComputes;
    /* we did that because we assign values and do read-writes in separate if statements, 
     * in between __syncwarp() calls */
    // this sum is only for molecule 1; therefore, drop 1 subscript; a,b,c denote O, H1, H2, respectively
    fs_sum_a = make_real3(0.0, 0.0, 0.0);
    fs_sum_b = make_real3(0.0, 0.0, 0.0);
    fs_sum_c = make_real3(0.0, 0.0, 0.0);

    // aggregate virials and forces for the O, H, H of the reference molecules within this warp
    if (COMP_VIRIALS) {
        virialsSum_a = Virial(0,0,0,0,0,0);
        virialsSum_b = Virial(0,0,0,0,0,0);
        virialsSum_c = Virial(0,0,0,0,0,0);
    }

    // this will be true or false for an entire warp
    if (moleculeIdx < nMolecules) {
        atomsReferenceMolecule = atomsFromMolecule[moleculeIdx]; // get our reference molecule atom idxs
        // load referencePos of atoms from global memory;
        real4 pos_a1_whole = xs[atomsReferenceMolecule.x];
        real4 pos_b1_whole = xs[atomsReferenceMolecule.y];
        real4 pos_c1_whole = xs[atomsReferenceMolecule.z];
        // get positions as real3
        pos_a1 = make_real3(pos_a1_whole);
        pos_b1 = make_real3(pos_b1_whole);
        pos_c1 = make_real3(pos_c1_whole);
        
        neighborlistSize = neighborCounts[moleculeIdx];
        //maxNumComputes = 0.5 * (neighborlistSize * (neighborlistSize - 1)); // number of unique triplets
        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        initNlistIdx= threadIdx.x % warpSize; // begins as 0...31
        int curNlistIdx = initNlistIdx;
        base_smem_idx = (threadIdx.x / warpSize) * maxNumNeighbors * 3; // this neighborlist begins at warpIdx * maxNumNeighbors * 3 (atoms perNeighbor)
        // these neighborlist idxs (of MOLECULE idxs!) is stored as warpIdx * maxNumNeighbors..
        // note that the pointer was already advanced beyond the shared memory for the positions.
        base_smem_idx_idxs = (threadIdx.x / warpSize) * (maxNumNeighbors+1);
        real3 xs_O, xs_H1, xs_H2;
        real4 xs_O_whole, xs_H1_whole, xs_H2_whole;
        // curNlistIdx goes as the set {0...31} + (32 * N), N = 0...ceil(numNeighbors/32)
        while (curNlistIdx < neighborlistSize) {
            // retrieve the neighborlist molecule idx corresponding to this thread
            int nlistIdx = baseIdx + initNlistIdx +  warpSize * (curNlistIdx/warpSize);
            int neighborMoleculeIdx = neighborlist[nlistIdx];                  // global memory access
            int4 neighborAtomIdxs    = atomsFromMolecule[neighborMoleculeIdx]; // global memory access

            // shared memory thus has a size of (3 * maxNumNeighbors * sizeof(real3)) -- we do not need M-site position for this potential
            int idx_in_smem_O = (3 * curNlistIdx) + base_smem_idx;                          // put H1, H2 directly after this O
            xs_O_whole = xs[neighborAtomIdxs.x]; // global memory access
            xs_H1_whole= xs[neighborAtomIdxs.y]; // global memory access
            xs_H2_whole= xs[neighborAtomIdxs.z]; // global memory access
            
            xs_O  = make_real3(xs_O_whole);
            xs_H1 = make_real3(xs_H1_whole);
            xs_H2 = make_real3(xs_H2_whole);
            
            smem_pos[idx_in_smem_O]   = xs_O;
            smem_pos[idx_in_smem_O+1] = xs_H1;
            smem_pos[idx_in_smem_O+2] = xs_H2;
            
            // and also stored the molecule idx; start at base_smem_idx_idxs, advance as curNlistIdx
            jMoleculeIdxs[base_smem_idx_idxs + curNlistIdx] = neighborMoleculeIdx;
            curNlistIdx += warpSize;                                        // advance as a warpSize
        }
     
        // finally, add myself to the end of the list (we check this list later when deciding on a 
        // k-molecule idx
        if (threadIdx.x % warpSize == 0) {
            jMoleculeIdxs[base_smem_idx_idxs + neighborlistSize] = moleculeIdx;
        }
    } // end (moleculeIdx < nMolecules)
    __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                  // --- this is cuda 9.0 function
                  //
 

    if (moleculeIdx < nMolecules) {    
        // ok, all neighboring atom positions are now stored sequentially in shared memory, grouped by neighboring molecule;
        /*
        int pair_pair_idx = initNlistIdx; // my index of the pair-pair computation, 0...31
        int reduced_idx   = pair_pair_idx;
        int jIdx          = 0;
        int kIdx          = jIdx + 1 + reduced_idx;
        int pair_computes_this_row = neighborlistSize - 1; // initialize
        */

        // looping over the triplets and storing the virial sum; have E3B evaluator take rij vectors, computed here
        real3 pos_a2,pos_b2,pos_c2;
        real3 pos_a3,pos_b3,pos_c3;
        real3 r_a1b2,r_a1c2,r_b1a2,r_c1a2;
        real3 r_a1b3,r_a1c3,r_b1a3,r_c1a3;
        real3 r_a2b3,r_a2c3,r_b2a3,r_c2a3;

        /*
        // here, all i,j,k triplets, where j and k both on i's neighborlist
        while (pair_pair_idx < maxNumComputes) {
            // pair_pair_idx is up-to-date; reduced_idx was incremented, but has not been reduced
            // jIdx, reduced_idx might not be;
            while (reduced_idx >= pair_computes_this_row) {
                // as long as pair_pair_idx < maxNumComputes, pair_computes_this_row is guaranteed 
                // to be larger than 0
                // ---- tested in python; for all values of pair_pair_idx [0...maxNumComputes),
                // gives unique pairs [j,k], k = [j+1....N-1], where N is neighborCount, and j is 0-indexed
                // increment jIdx
                jIdx++;
                // reduce the reduced_idx by pair_computes_this_row
                reduced_idx -= pair_computes_this_row;
                // the pair_computes_this_row decreases by 1
                pair_computes_this_row--;
                // our kIdx is now as (jIdx + 1) + reduced_idx;
                // --- reduced idx is now the number of advances /after/ jIdx + 1
                kIdx = jIdx + 1 + reduced_idx;
            }
            // we have our 'i' molecule in the i-j-k triplet;
            // pair_pair_idx specifies where in the j, k flattened array we are
            // --- we need to compute what j is, and then get k; both will be obtained from shared memory

            // we now have kIdx, jIdx;
            // load atom positions from shared memory
            // --- account for this molecules' displacement in smem for the block! 
            //     so, increment by base_smem_idx
            pos_a2 = smem_pos[3*jIdx     + base_smem_idx];
            pos_b2 = smem_pos[3*jIdx + 1 + base_smem_idx];
            pos_c2 = smem_pos[3*jIdx + 2 + base_smem_idx];
            
            pos_a3 = smem_pos[3*kIdx     + base_smem_idx];
            pos_b3 = smem_pos[3*kIdx + 1 + base_smem_idx];
            pos_c3 = smem_pos[3*kIdx + 2 + base_smem_idx];

            // compute the 12 unique vectors for this triplet; pass these, and the force, and virials to evaluator
            // -- no possible way to put these in to shared memory; here, we must do redundant calculations.

            // rij = ri - rj

            // i = 1, j = 2
            // rij: i = a1, j = b2
            r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
            // rij: i = a1, j = c2
            r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
            // rij: i = b1, j = a2
            r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
            // rij: i = c1, j = a2
            r_c1a2 = bounds.minImage(pos_c1 - pos_a2);
            
            // i = 1, j = 3
            // rij: i = a1, j = b3
            r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
            // rij: i = a1, j = c3
            r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
            // rij: i = b1, j = a3
            r_b1a3 = bounds.minImage(pos_b1 - pos_a3);
            // rij: i = c1, j = a3
            r_c1a3 = bounds.minImage(pos_c1 - pos_a3);

            // i = 2, j = 3
            // rij: i = a2, j = b3
            r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
            // rij: i = a2, j = c3
            r_a2c3 = bounds.minImage(pos_a2 - pos_c3);
            // rij: i = b2, j = a3
            r_b2a3 = bounds.minImage(pos_b2 - pos_a3);
            // rij: i = c2, j = a3
            r_c2a3 = bounds.minImage(pos_c2 - pos_a3);

            // send distance vectors, force sums, and virials to evaluator (guaranteed to be e3b type)
            if (COMP_VIRIALS) {
                eval.threeBodyForce<true>(fs_sum_a, fs_sum_b, fs_sum_c,
                                 virialsSum_a, virialsSum_b, virialsSum_c,
                                 r_a1b2, r_a1c2, r_b1a2, r_c1a2,
                                 r_a1b3, r_a1c3, r_b1a3, r_c1a3,
                                 r_a2b3, r_a2c3, r_b2a3, r_c2a3);
            } else {
                eval.threeBodyForce<false>(fs_sum_a, fs_sum_b, fs_sum_c,
                                 virialsSum_a, virialsSum_b, virialsSum_c,
                                 r_a1b2, r_a1c2, r_b1a2, r_c1a2,
                                 r_a1b3, r_a1c3, r_b1a3, r_c1a3,
                                 r_a2b3, r_a2c3, r_b2a3, r_c2a3);
            }
            
            pair_pair_idx += warpSize; // advance pair_pair_idx through the nlist by warpSize
            reduced_idx   += warpSize; // advanced reduced_idx as warpSize; will be reduced at next loop
        }
        */


        // let's divide the j's into groups of 4..
        // initially, a thread starts on j = 0, 1, 2 or 3...
        int jWorkGroupSize = 4; // four threads to a j
        int idxInWarp = threadIdx.x % warpSize; // 0...31
        int thisThreadjIdxInit = (idxInWarp) / jWorkGroupSize; // 0...7
        int workGroupsPerWarp = warpSize / jWorkGroupSize;
        int idxInWorkGroup = idxInWarp % jWorkGroupSize; // 0,1,2,3 ...

        // ok, so we have 8 groups of threads (4 threads in a group) advancing over the j neighborlist;
        // they are incremented by 8 at the conclusion of the computation
        // --- the kIdxs are traversed by the 4 threads within a given workgroup, advanced by workGroupSize (4)
        for (int jIdx = thisThreadjIdxInit; jIdx < neighborlistSize; jIdx+=workGroupsPerWarp) {
            for (int kIdx = idxInWorkGroup; kIdx < neighborlistSize; kIdx+=jWorkGroupSize) {
                if (kIdx == jIdx) continue;
                pos_a2 = smem_pos[3*jIdx     + base_smem_idx];
                pos_b2 = smem_pos[3*jIdx + 1 + base_smem_idx];
                pos_c2 = smem_pos[3*jIdx + 2 + base_smem_idx];
                
                pos_a3 = smem_pos[3*kIdx     + base_smem_idx];
                pos_b3 = smem_pos[3*kIdx + 1 + base_smem_idx];
                pos_c3 = smem_pos[3*kIdx + 2 + base_smem_idx];

                // compute the 12 unique vectors for this triplet; pass these, and the force, and virials to evaluator
                // -- no possible way to put these in to shared memory; here, we must do redundant calculations.

                // rij = ri - rj

                // i = 1, j = 2
                // rij: i = a1, j = b2
                r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
                // rij: i = a1, j = c2
                r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
                // rij: i = b1, j = a2
                r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
                // rij: i = c1, j = a2
                r_c1a2 = bounds.minImage(pos_c1 - pos_a2);
                
                // i = 1, j = 3
                // rij: i = a1, j = b3
                r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
                // rij: i = a1, j = c3
                r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
                // rij: i = b1, j = a3
                r_b1a3 = bounds.minImage(pos_b1 - pos_a3);
                // rij: i = c1, j = a3
                r_c1a3 = bounds.minImage(pos_c1 - pos_a3);

                // i = 2, j = 3
                // rij: i = a2, j = b3
                r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
                // rij: i = a2, j = c3
                r_a2c3 = bounds.minImage(pos_a2 - pos_c3);
                // rij: i = b2, j = a3
                r_b2a3 = bounds.minImage(pos_b2 - pos_a3);
                // rij: i = c2, j = a3
                r_c2a3 = bounds.minImage(pos_c2 - pos_a3);

                // send distance vectors, force sums, and virials to evaluator (guaranteed to be e3b type)
                if (COMP_VIRIALS) {
                    eval.threeBodyForce<true>(fs_sum_a, fs_sum_b, fs_sum_c,
                                     virialsSum_a, virialsSum_b, virialsSum_c,
                                     r_a1b2, r_a1c2, r_b1a2, r_c1a2,
                                     r_a1b3, r_a1c3, r_b1a3, r_c1a3,
                                     r_a2b3, r_a2c3, r_b2a3, r_c2a3);
                } else {
                    eval.threeBodyForce<false>(fs_sum_a, fs_sum_b, fs_sum_c,
                                     virialsSum_a, virialsSum_b, virialsSum_c,
                                     r_a1b2, r_a1c2, r_b1a2, r_c1a2,
                                     r_a1b3, r_a1c3, r_b1a3, r_c1a3,
                                     r_a2b3, r_a2c3, r_b2a3, r_c2a3);
                }
            } // kIdx
        } // jIdx
    }

    __syncwarp();
    
    // we need to wait until all threads are done computing their forces (and virials)
    // now, do lane shifting to accumulate the forces (and virials, if necessary) in to threadIdx == 0 which does global write
    // warpReduce all forces; if virials, warpReduce all Virials as well
    // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
    agg_fs_sum_a = warpReduceSum(fs_sum_a,warpSize);
    agg_fs_sum_b = warpReduceSum(fs_sum_b,warpSize);
    agg_fs_sum_c = warpReduceSum(fs_sum_c,warpSize);

    //if (threadIdx.x % warpSize == 0) {
    //    printf("fs_sum_a: %f %f %f\n", fs_sum_a.x, fs_sum_a.y, fs_sum_a.z);
    //}
    if (COMP_VIRIALS) {
        // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
        // virials for oxygen of reference molecule
        agg_virialsSum_a = warpReduceSum(virialsSum_a,warpSize);
        // virials for hydrogen H1 of reference molecule
        agg_virialsSum_b = warpReduceSum(virialsSum_b,warpSize);
        // virials for hydrogen H2 of reference molecule
        agg_virialsSum_c = warpReduceSum(virialsSum_c,warpSize);
    
    }
    // no syncing required after warp reductions
    
    // threadIdx.x % warpSize does global write, iff. moleculeIdx < nMolecules
    if (((threadIdx.x % warpSize) == 0) and (moleculeIdx < nMolecules)) {
        // load curForce on O molecule
        real4 curForce_O = fs[atomsReferenceMolecule.x];
        real4 curForce_H1= fs[atomsReferenceMolecule.y];
        real4 curForce_H2= fs[atomsReferenceMolecule.z];

        // add contributions from E3B
        curForce_O += agg_fs_sum_a;
        curForce_H1+= agg_fs_sum_b;
        curForce_H2+= agg_fs_sum_c;

        if (COMP_VIRIALS) {
            // load from global memory
            Virial virial_O = virials[atomsReferenceMolecule.x];
            Virial virial_H1= virials[atomsReferenceMolecule.y];
            Virial virial_H2= virials[atomsReferenceMolecule.z];
            // add contributions from E3B to global value
            virial_O += agg_virialsSum_a;
            virial_H1+= agg_virialsSum_b;
            virial_H2+= agg_virialsSum_c;
            // write to global memory
            virials[atomsReferenceMolecule.x] = virial_O;
            virials[atomsReferenceMolecule.y] = virial_H1;
            virials[atomsReferenceMolecule.z] = virial_H2;
        }
        // write forces to global
        fs[atomsReferenceMolecule.x] = curForce_O;
        fs[atomsReferenceMolecule.y] = curForce_H1;
        fs[atomsReferenceMolecule.z] = curForce_H2;
    }

} // end kernel


// computes the i-j-k triplets, k not on i's neighborlist
template <bool COMP_VIRIALS> 
__global__ void compute_E3B_force_edge
        (int nMolecules, 
         const int4 *__restrict__ atomsFromMolecule,  // accessed by molecule idx, as atomidxs
         const uint16_t *__restrict__ neighborCounts, // by moleculeIdx
         const uint *__restrict__ neighborlist,       // bymoleculeIdxs
         const uint32_t * __restrict__ cumulSumMaxPerBlock, // gridGPULocal
         int warpSize, // device property
         const real4 *__restrict__ xs,  // as atom idxs
         real4 *__restrict__ fs,        // as atom idxs
         BoundsGPU bounds, 
         Virial *__restrict__ virials,  // as atom idxs
         int nMoleculesPerBlock,        
         int maxNumNeighbors,
         EvaluatorE3B eval)
{

    // we have one molecule per warp, presumably;
    // store neighborlist as atomIdxsInMolecule
    extern __shared__ real3 smem_neighborAtomPos[];

    real3 *smem_pos = smem_neighborAtomPos;
    // my shared memory for this block will be as atom positions, by reference molecule... then jMoleculeIdxs, by reference molecule
    uint32_t *jMoleculeIdxs = (uint32_t*)&smem_pos[3*maxNumNeighbors*nMoleculesPerBlock]; 

    // populate smem_neighborIdxs for this warp
    // blockDim.x gives number of threads in a block in x direction
    // gridDim.x gives number of blocks in a grid in the x direction
    // blockDim.x * gridDim.x gives n threads in grid in x direction
    // blockIdx.x * (nMoleculesPerBlock) : e.g., blockIdx.x == 0: 0... then threadIdx.x/warpSize: 0...7
    //        so first 8 molecules
    // blockIdx.x == 1 --> moleculeIdx = [8,..,15], etc. OK.
    // ---- blockIdx.x * (nMoleculesPerBlock) + (threadIdx.x /32) gives moleculeIdx
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // get where our neighborlist for this molecule starts // TODO verify
    //int baseIdx = baseNeighlistIdxFromIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx);
    // baseIdx is as if we pass cumulSumMaxPerBlock array, warpSize... then atomIdx is moleculeIdx, 
    // nThreadPerAtom ~ warpSize; this gives our starting point in nlist.
    int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx, warpSize);

    /* declare the variables */
    Virial virialsSum_a;
    Virial virialsSum_b;
    Virial virialsSum_c;
    real3 fs_sum_a, fs_sum_b, fs_sum_c;

    Virial agg_virialsSum_a, agg_virialsSum_b, agg_virialsSum_c;
    real3 agg_fs_sum_a, agg_fs_sum_b, agg_fs_sum_c;

    int4 atomsReferenceMolecule;
    real3 pos_a1,pos_b1,pos_c1;
    int neighborlistSize, base_smem_idx,initNlistIdx,base_smem_idx_idxs;
    /* we did that because we assign values and do read-writes in separate if statements, 
     * in between __syncwarp() calls */
    // this sum is only for molecule 1; therefore, drop 1 subscript; a,b,c denote O, H1, H2, respectively
    fs_sum_a = make_real3(0.0, 0.0, 0.0);
    fs_sum_b = make_real3(0.0, 0.0, 0.0);
    fs_sum_c = make_real3(0.0, 0.0, 0.0);

    // aggregate virials and forces for the O, H, H of the reference molecules within this warp
    if (COMP_VIRIALS) {
        virialsSum_a = Virial(0,0,0,0,0,0);
        virialsSum_b = Virial(0,0,0,0,0,0);
        virialsSum_c = Virial(0,0,0,0,0,0);
    }

    // this will be true or false for an entire warp
    if (moleculeIdx < nMolecules) {
        atomsReferenceMolecule = atomsFromMolecule[moleculeIdx]; // get our reference molecule atom idxs
        // load referencePos of atoms from global memory;
        real4 pos_a1_whole = xs[atomsReferenceMolecule.x];
        real4 pos_b1_whole = xs[atomsReferenceMolecule.y];
        real4 pos_c1_whole = xs[atomsReferenceMolecule.z];
        // get positions as real3
        pos_a1 = make_real3(pos_a1_whole);
        pos_b1 = make_real3(pos_b1_whole);
        pos_c1 = make_real3(pos_c1_whole);
        
        neighborlistSize = neighborCounts[moleculeIdx];
        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        initNlistIdx= threadIdx.x % warpSize; // begins as 0...31
        int curNlistIdx = initNlistIdx;
        base_smem_idx = (threadIdx.x / warpSize) * maxNumNeighbors * 3; // this neighborlist begins at warpIdx * maxNumNeighbors * 3 (atoms perNeighbor)
        // these neighborlist idxs (of MOLECULE idxs!) is stored as warpIdx * maxNumNeighbors..
        // note that the pointer was already advanced beyond the shared memory for the positions.
        base_smem_idx_idxs = (threadIdx.x / warpSize) * (maxNumNeighbors+1);
        real3 xs_O, xs_H1, xs_H2;
        real4 xs_O_whole, xs_H1_whole, xs_H2_whole;
        // curNlistIdx goes as the set {0...31} + (32 * N), N = 0...ceil(numNeighbors/32)
        while (curNlistIdx < neighborlistSize) {
            // retrieve the neighborlist molecule idx corresponding to this thread
            int nlistIdx = baseIdx + initNlistIdx +  warpSize * (curNlistIdx/warpSize);
            int neighborMoleculeIdx = neighborlist[nlistIdx];                  // global memory access
            int4 neighborAtomIdxs    = atomsFromMolecule[neighborMoleculeIdx]; // global memory access

            // shared memory thus has a size of (3 * maxNumNeighbors * sizeof(real3)) -- we do not need M-site position for this potential
            int idx_in_smem_O = (3 * curNlistIdx) + base_smem_idx;                          // put H1, H2 directly after this O
            xs_O_whole = xs[neighborAtomIdxs.x]; // global memory access
            xs_H1_whole= xs[neighborAtomIdxs.y]; // global memory access
            xs_H2_whole= xs[neighborAtomIdxs.z]; // global memory access
            
            xs_O  = make_real3(xs_O_whole);
            xs_H1 = make_real3(xs_H1_whole);
            xs_H2 = make_real3(xs_H2_whole);
            
            smem_pos[idx_in_smem_O]   = xs_O;
            smem_pos[idx_in_smem_O+1] = xs_H1;
            smem_pos[idx_in_smem_O+2] = xs_H2;
            
            // and also stored the molecule idx; start at base_smem_idx_idxs, advance as curNlistIdx
            jMoleculeIdxs[base_smem_idx_idxs + curNlistIdx] = neighborMoleculeIdx;

            curNlistIdx += warpSize;                                        // advance as a warpSize
        }
     
        // finally, add myself to the end of the list (we check this list later when deciding on a 
        // k-molecule idx
        if (threadIdx.x % warpSize == 0) {
            jMoleculeIdxs[base_smem_idx_idxs + neighborlistSize] = moleculeIdx;
        }
    } // end (moleculeIdx < nMolecules)
    __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                  // --- this is cuda 9.0 function
                  //
 
    // and now, do (i,j), k on j's nlist, k not on i's nlist 
    // -- we have the global molecule idxs on i's list in smem;
    //    simply check the index against the entries in the list.  
    //    if not in list, do (i,j,k) compute
    if (moleculeIdx < nMolecules) {    
        // ok, so, as a warp, we pick a j; then, a given threadIdx.x % warpSize gives its idx in j's nlist; while 
        // idx in j's nlist < j's neighbor counts, continue; then, advance j until we have computed all j's
        // --- for each j, skip (i,j,k) if k is on i's neighborlist (i.e., if
        real3 pos_a2,pos_b2,pos_c2;
        real3 pos_a3,pos_b3,pos_c3;
        real3 r_a1b2,r_a1c2,r_b1a2,r_c1a2;
        real3 r_a1b3,r_a1c3,r_b1a3,r_c1a3;
        real3 r_a2b3,r_a2c3,r_b2a3,r_c2a3;
        real4 xs_O_whole, xs_H1_whole, xs_H2_whole;
       

        // it would probably be faster to split the 32 threads in to 8 groups of 4 that 
        //       work on a different j per group of 8, with 4 threads iterating over each j's neighbors
        // -- we could do any combination of work group sizes s.t. 32
        int workGroupSize = 4;
        int warpIdxInBlock  = threadIdx.x % warpSize; // 0...31
        int workGroupIdx    = warpIdxInBlock / 4;   // 0...7
        int kIdxInWorkGroup = warpIdxInBlock % 4;    // 0...3
        int nlistIncrement  = warpSize / 4; // warpSize / workGroupSize  
        // ok, so workGroupIdx will determine which jIdx we are working on;
        // kIdxInWorkGroup will determine which kIdx in a workgroup we are working on

        // LOOPING OVER MOLECULE 'i' NEIGHBORLIST TO GET J MOLECULE
        // get j index from global nlist, and load position from smem
        for (int jIdx = workGroupIdx; jIdx < neighborlistSize; jIdx+=nlistIncrement) {

            // load j's positions; we'll definitely need these.
            // jIdx as 0... neighborlistSize provides easy access to i's neighborlist in smem.
            // --- these do not change except when j loop is indexed
            pos_a2 = smem_pos[3*jIdx     + base_smem_idx];
            pos_b2 = smem_pos[3*jIdx + 1 + base_smem_idx];
            pos_c2 = smem_pos[3*jIdx + 2 + base_smem_idx];


            // compute the rij vectors here - they only change when j changes.
            // rij: i = a1, j = b2
            r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
            // rij: i = a1, j = c2
            r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
            // rij: i = b1, j = a2
            r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
            // rij: i = c1, j = a2
            r_c1a2 = bounds.minImage(pos_c1 - pos_a2);


            // we stored these in shared memory the first time we read them in
            int globalJIdx = jMoleculeIdxs[base_smem_idx_idxs + jIdx];
            // so, i need j's global molecule idx to get its neighborlist - global memory access
            // --- also, because each thread is accessing a constant j per 'for' loop, we do not want 
            //     the initNlistIdx here (next line copied from above; modified the line below it)
            
            // ok, all threads in a warp now have the globalJIdx; now, we iterate as a warp over j's neighborlist 
            // to get K, with checking that k not on i's neighborlist
            
            // actually, because of padding on neighborlists etc... and variability that might be incurred by 
            // modifying nlist size to permit fewer nlist computations etc., it is probably easiest to store
            // the idxs in a linear array in smem, and simply check that kIdx not in array.

            // getting j molecule's neighborcounts
            int jNeighCounts = neighborCounts[globalJIdx];

            // load where j molecule idx's neighborlist starts in global memory, using its global index
            int baseIdx_globalJIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, globalJIdx, warpSize);
           
            int nextOffset = 0; // because we traverse these in groups of 4, not nAtomsPerThread
            int k_idx_j_neighbors = kIdxInWorkGroup; // as 0...3
            int initNlistIdx_jNlist = k_idx_j_neighbors;
            while ((k_idx_j_neighbors + nextOffset) < jNeighCounts) {
                // get k idx from the actual j neighborlist - first, advance nlistIdx
                bool compute_ijk = true; // default to true; if we find k on i's nlist, set to false; then, check boolean

                int nlistIdx_jNlist = baseIdx_globalJIdx + initNlistIdx_jNlist + nextOffset + warpSize * (k_idx_j_neighbors/warpSize);
                // this is our k molecule idx
                int kIdx = neighborlist[nlistIdx_jNlist];                  // global memory access
                
                nextOffset += workGroupSize;
                
                if (! (nextOffset % warpSize) ) {
                    nextOffset = 0;
                    k_idx_j_neighbors += warpSize;
                }
                // check if k is on i's neighborlist, or is i itself; this amounts to checking ~ 30 integers
                // the good thing is: this should be broadcast to all threads, so no serialized access!
                // --- neighborlistSize here is the size of i's neighborlist; recall that in shared memory we also 
                //     appended i's moleculeIdx at the end of this list.
                for (int idxs_to_check=0; idxs_to_check < neighborlistSize+1; idxs_to_check++) {
                    int idx_on_i_nlist = jMoleculeIdxs[base_smem_idx_idxs + idxs_to_check];
                    if (kIdx == idx_on_i_nlist) compute_ijk = false;
                }

                if (compute_ijk) {
                    // load k positions from global memory and do the computation.
                    int4 neighborAtomIdxs    = atomsFromMolecule[kIdx]; // global memory access

                    xs_O_whole = xs[neighborAtomIdxs.x]; // global memory access
                    xs_H1_whole= xs[neighborAtomIdxs.y]; // global memory access
                    xs_H2_whole= xs[neighborAtomIdxs.z]; // global memory access
                    
                    pos_a3 = make_real3(xs_O_whole);
                    pos_b3 = make_real3(xs_H1_whole);
                    pos_c3 = make_real3(xs_H2_whole);

                    // rij = ri - rj

                    // i = 1, j = 3
                    //
                    // rij: i = a1, j = b3
                    r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
                    // rij: i = a1, j = c3
                    r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
                    // rij: i = b1, j = a3
                    r_b1a3 = bounds.minImage(pos_b1 - pos_a3);
                    // rij: i = c1, j = a3
                    r_c1a3 = bounds.minImage(pos_c1 - pos_a3);

                    // i = 2, j = 3
                    //
                    // rij: i = a2, j = b3
                    r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
                    // rij: i = a2, j = c3
                    r_a2c3 = bounds.minImage(pos_a2 - pos_c3);
                    // rij: i = b2, j = a3
                    r_b2a3 = bounds.minImage(pos_b2 - pos_a3);
                    // rij: i = c2, j = a3
                    r_c2a3 = bounds.minImage(pos_c2 - pos_a3);

                    // send distance vectors, force sums, and virials to evaluator (guaranteed to be e3b type)
                    if (COMP_VIRIALS) {
                        eval.threeBodyForce_edge<true>(fs_sum_a, fs_sum_b, fs_sum_c,
                                                       virialsSum_a, virialsSum_b, virialsSum_c,
                                                       r_a1b2, r_a1c2, r_b1a2, r_c1a2,
                                                       r_a1b3, r_a1c3, r_b1a3, r_c1a3,
                                                       r_a2b3, r_a2c3, r_b2a3, r_c2a3);
                    } else {
                        eval.threeBodyForce_edge<false>(fs_sum_a, fs_sum_b, fs_sum_c,
                                                        virialsSum_a, virialsSum_b, virialsSum_c,
                                                        r_a1b2, r_a1c2, r_b1a2, r_c1a2,
                                                        r_a1b3, r_a1c3, r_b1a3, r_c1a3,
                                                        r_a2b3, r_a2c3, r_b2a3, r_c2a3);
                    }
                } // end if compute_ijk

                // finally, advance k_idx_j_neighbors as warpSize
                //k_idx_j_neighbors += workGroupSize;
            } // end loop over j's neighborlist
        } // end loop over i's neighborlist
    }

    __syncwarp();
    
    // we need to wait until all threads are done computing their forces (and virials)
    // now, do lane shifting to accumulate the forces (and virials, if necessary) in to threadIdx == 0 which does global write
    // warpReduce all forces; if virials, warpReduce all Virials as well
    // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
    agg_fs_sum_a = warpReduceSum(fs_sum_a,warpSize);
    agg_fs_sum_b = warpReduceSum(fs_sum_b,warpSize);
    agg_fs_sum_c = warpReduceSum(fs_sum_c,warpSize);
    if (COMP_VIRIALS) {
        // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
        // virials for oxygen of reference molecule
        agg_virialsSum_a = warpReduceSum(virialsSum_a,warpSize);
        // virials for hydrogen H1 of reference molecule
        agg_virialsSum_b = warpReduceSum(virialsSum_b,warpSize);
        // virials for hydrogen H2 of reference molecule
        agg_virialsSum_c = warpReduceSum(virialsSum_c,warpSize);
    
    }
    // no syncing required after warp reductions
    
    // threadIdx.x % warpSize does global write, iff. moleculeIdx < nMolecules
    if (((threadIdx.x % warpSize) == 0) and (moleculeIdx < nMolecules)) {
        // load curForce on O molecule
        real4 curForce_O = fs[atomsReferenceMolecule.x];
        real4 curForce_H1= fs[atomsReferenceMolecule.y];
        real4 curForce_H2= fs[atomsReferenceMolecule.z];

        // add contributions from E3B
        curForce_O += agg_fs_sum_a;
        curForce_H1+= agg_fs_sum_b;
        curForce_H2+= agg_fs_sum_c;

        if (COMP_VIRIALS) {
            // load from global memory
            Virial virial_O = virials[atomsReferenceMolecule.x];
            Virial virial_H1= virials[atomsReferenceMolecule.y];
            Virial virial_H2= virials[atomsReferenceMolecule.z];
            // add contributions from E3B to global value
            virial_O += agg_virialsSum_a;
            virial_H1+= agg_virialsSum_b;
            virial_H2+= agg_virialsSum_c;
            // write to global memory
            virials[atomsReferenceMolecule.x] = virial_O;
            virials[atomsReferenceMolecule.y] = virial_H1;
            virials[atomsReferenceMolecule.z] = virial_H2;
        }
        // write forces to global
        fs[atomsReferenceMolecule.x] = curForce_O;
        fs[atomsReferenceMolecule.y] = curForce_H1;
        fs[atomsReferenceMolecule.z] = curForce_H2;
    }

} // end kernel

  
// compute the twobody energy for e3b
__global__ void compute_E3B_energy_twobody
        (int nMolecules, 
         const int4 *__restrict__ atomsFromMolecule,         // by moleculeIdx, has atomIdxs
         const uint16_t *__restrict__ neighborCounts,        // gridGPULocal
         const uint *__restrict__ neighborlist,              // gridGPULocal
         const uint32_t * __restrict__ cumulSumMaxPerBlock,  // gridGPULocal
         int warpSize, 
         const real4 *__restrict__ xs,                       // as atomIdxs
         real * __restrict__ perParticleEng,                 // this is per-particle (O,H,H)
         BoundsGPU bounds, 
         int nMoleculesPerBlock,
         int maxNumNeighbors,
         EvaluatorE3B eval)
{

    // we have one molecule per warp, presumably;
    // store neighborlist as atomIdxsInMolecule
    
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // get where our neighborlist for this molecule starts
    int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx,warpSize);

    //real eng_sum_a, eng_sum_b, eng_sum_c;
    real pair_eng_sum_a;
    int4 atomsReferenceMolecule;
    real3 pos_a1; //, pos_b1, pos_c1;
    real3 pos_a2;//, pos_b2, pos_c2;
    //real3 pos_a3, pos_b3, pos_c3;
    int neighborlistSize,initNlistIdx;
    // this will be true or false for an entire warp
    if (moleculeIdx < nMolecules) {

        pair_eng_sum_a = 0.0;
        /* NOTE to others: see the notation used in 
         * Kumar and Skinner, J. Phys. Chem. B., 2008, 112, 8311-8318
         * "Water Simulation Model with Explicit 3 Body Interactions"
         *
         * we use their notation for decomposing the molecules into constituent atoms a,b,c (oxygen, hydrogen, hydrogen)
         * and decomposing the given trimer into the set of molecules 1,2,3 (water molecule 1, 2, and 3)
         */
        atomsReferenceMolecule = atomsFromMolecule[moleculeIdx]; // get our reference molecule atom idxs
        
        // load referencePos of atoms from global memory;
        real4 pos_a1_whole = xs[atomsReferenceMolecule.x];
        // get positions as real3
        pos_a1 = make_real3(pos_a1_whole);
        
        neighborlistSize = neighborCounts[moleculeIdx];
        initNlistIdx = threadIdx.x % warpSize; // begins as 0...31
        int curNlistIdx = initNlistIdx;
        real4 xs_O_whole;//, xs_H1_whole, xs_H2_whole;
        
        // curNlistIdx goes as the set {0...31} + (32 * N), N = 0...ceil(numNeighbors/32)
        while (curNlistIdx < neighborlistSize) {
            // retrieve the neighborlist molecule idx corresponding to this thread
            int nlistIdx = baseIdx + initNlistIdx +  warpSize * (curNlistIdx/warpSize);
            int neighborMoleculeIdx = neighborlist[nlistIdx];                  // global memory access
            int4 neighborAtomIdxs    = atomsFromMolecule[neighborMoleculeIdx]; // global memory access
            
            xs_O_whole = xs[neighborAtomIdxs.x]; // global memory access
            //xs_H1_whole= xs[neighborAtomIdxs.y]; // global memory access
            //xs_H2_whole= xs[neighborAtomIdxs.z]; // global memory access
            
            pos_a2  = make_real3(xs_O_whole);
            
            // and also stored the molecule idx; start at base_smem_idx_idxs, advance as curNlistIdx
            //jMoleculeIdxs[base_smem_idx_idxs + curNlistIdx] = neighborMoleculeIdx;

            real3 rij_a1a2 = bounds.minImage(pos_a1 - pos_a2);
            real rij_a1a2_scalar = length(rij_a1a2);
            pair_eng_sum_a += 0.5 * eval.twoBodyEnergy(rij_a1a2_scalar); // 0.5 because we double count all two body pairs
            curNlistIdx += warpSize;                                        // advance as a warpSize
        }
    }
    __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                  // --- this is cuda 9.0 function; only needed for Volta architectures (and later)
   
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        pair_eng_sum_a += __shfl_down_sync(0xffffffff,pair_eng_sum_a, offset);
    }
    
    // threadIdx.x % warpSize  == 0 does global write
    if (((threadIdx.x % warpSize) == 0) and (moleculeIdx < nMolecules)) {
        // load current energies on O, H, H of reference molecule
        real cur_eng_O  = perParticleEng[atomsReferenceMolecule.x];
        
        cur_eng_O += pair_eng_sum_a;
        // write energies to global
        perParticleEng[atomsReferenceMolecule.x] = cur_eng_O;
    }

} // end compute_E3B_energy


// compute e3b energy for triplet i-j-k, j and k on i's neighborlist
__global__ void compute_E3B_energy_center
        (int nMolecules, 
         const int4 *__restrict__ atomsFromMolecule,         // by moleculeIdx, has atomIdxs
         const uint16_t *__restrict__ neighborCounts,        // gridGPULocal
         const uint *__restrict__ neighborlist,              // gridGPULocal
         const uint32_t * __restrict__ cumulSumMaxPerBlock,  // gridGPULocal
         int warpSize, 
         const real4 *__restrict__ xs,                       // as atomIdxs
         real * __restrict__ perParticleEng,                 // this is per-particle (O,H,H)
         BoundsGPU bounds, 
         int nMoleculesPerBlock,
         int maxNumNeighbors,
         EvaluatorE3B eval)
{

    // we have one molecule per warp, presumably;
    // store neighborlist as atomIdxsInMolecule
    extern __shared__ real3 smem_neighborAtomPos[];
    
    real3 *smem_pos = smem_neighborAtomPos;
    // my shared memory for this block will be as atom positions, by reference molecule... then jMoleculeIdxs, by reference molecule
    uint32_t *jMoleculeIdxs = (uint32_t*)&smem_pos[3*maxNumNeighbors*nMoleculesPerBlock]; 
    
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // get where our neighborlist for this molecule starts
    int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx,warpSize);

    real eng_sum_a, eng_sum_b, eng_sum_c;
    real3 agg_eng_sum;
    int4 atomsReferenceMolecule;
    real3 pos_a1, pos_b1, pos_c1;
    real3 pos_a2, pos_b2, pos_c2;
    real3 pos_a3, pos_b3, pos_c3;
    int neighborlistSize,base_smem_idx,initNlistIdx,base_smem_idx_idxs;
    //int maxNumComputes;
    // this will be true or false for an entire warp
    if (moleculeIdx < nMolecules) {

        // this sum is only for molecule 1; therefore, drop 1 subscript; a,b,c denote O, H1, H2, respectively
        eng_sum_a = 0.0;
        eng_sum_b = 0.0;
        eng_sum_c = 0.0;
        /* NOTE to others: see the notation used in 
         * Kumar and Skinner, J. Phys. Chem. B., 2008, 112, 8311-8318
         * "Water Simulation Model with Explicit 3 Body Interactions"
         *
         * we use their notation for decomposing the molecules into constituent atoms a,b,c (oxygen, hydrogen, hydrogen)
         * and decomposing the given trimer into the set of molecules 1,2,3 (water molecule 1, 2, and 3)
         */
        atomsReferenceMolecule = atomsFromMolecule[moleculeIdx]; // get our reference molecule atom idxs
        
        // load referencePos of atoms from global memory;
        real4 pos_a1_whole = xs[atomsReferenceMolecule.x];
        real4 pos_b1_whole = xs[atomsReferenceMolecule.y];
        real4 pos_c1_whole = xs[atomsReferenceMolecule.z];

        // get positions as real3
        pos_a1 = make_real3(pos_a1_whole);
        pos_b1 = make_real3(pos_b1_whole);
        pos_c1 = make_real3(pos_c1_whole);
        
        neighborlistSize = neighborCounts[moleculeIdx];
        //maxNumComputes = 0.5 * (neighborlistSize * (neighborlistSize - 1)); // number of unique triplets
        //if (threadIdx.x % 32 == 0) printf("moleculeIdx %d neighborlistSize %d\n",moleculeIdx,neighborlistSize);
        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        initNlistIdx= threadIdx.x % warpSize; // begins as 0...31
        int curNlistIdx = initNlistIdx;
        base_smem_idx = (threadIdx.x / warpSize) * maxNumNeighbors * 3; // this neighborlist begins at warpIdx * maxNumNeighbors * 3 (atoms perNeighbor)
        base_smem_idx_idxs = (threadIdx.x / warpSize) * (maxNumNeighbors+1);
        real4 xs_O_whole, xs_H1_whole, xs_H2_whole;
        /* LOAD NEIGHBOR ATOM POSITIONS IN TO SHARED MEMORY */
        // curNlistIdx goes as the set {0...31} + (32 * N), N = 0...ceil(numNeighbors/32)
        while (curNlistIdx < neighborlistSize) {
            // retrieve the neighborlist molecule idx corresponding to this thread
            int nlistIdx = baseIdx + initNlistIdx +  warpSize * (curNlistIdx/warpSize);
            int neighborMoleculeIdx = neighborlist[nlistIdx];                  // global memory access
            int4 neighborAtomIdxs    = atomsFromMolecule[neighborMoleculeIdx]; // global memory access
            
            // shared memory thus has a size of (3 * maxNumNeighbors * sizeof(real4)) -- we do not need M-site position for this potential
            int idx_in_smem_O = (3 * curNlistIdx) + base_smem_idx;                          // put H1, H2 directly after this O; base_smem_idx is by warpIdx
            xs_O_whole = xs[neighborAtomIdxs.x]; // global memory access
            xs_H1_whole= xs[neighborAtomIdxs.y]; // global memory access
            xs_H2_whole= xs[neighborAtomIdxs.z]; // global memory access
            
            pos_a2  = make_real3(xs_O_whole);
            pos_b2  = make_real3(xs_H1_whole);
            pos_c2  = make_real3(xs_H2_whole);
            
            smem_pos[idx_in_smem_O]   = pos_a2;
            smem_pos[idx_in_smem_O+1] = pos_b2;
            smem_pos[idx_in_smem_O+2] = pos_c2;
            
            // and also stored the molecule idx; start at base_smem_idx_idxs, advance as curNlistIdx
            jMoleculeIdxs[base_smem_idx_idxs + curNlistIdx] = neighborMoleculeIdx;

            curNlistIdx += warpSize;                                        // advance as a warpSize
        }

        if (threadIdx.x % warpSize == 0) {
            jMoleculeIdxs[base_smem_idx_idxs + neighborlistSize] = moleculeIdx;
        }
    }
    __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                  // --- this is cuda 9.0 function; only needed for Volta architectures (and later)
      
    real3 eng_sum_as_real3 = make_real3(0.0, 0.0, 0.0);
    if (moleculeIdx < nMolecules) {
        // ok, all neighboring atom positions are now stored sequentially in shared memory, grouped by neighboring molecule;
        /*
        int pair_pair_idx = initNlistIdx; // my index of the pair-pair computation, 0...31; i.e., threadIdx.x % warpSize
        int reduced_idx   = pair_pair_idx; // initialize reduced_idx as pair_pair_idx
        int jIdx          = 0; // initialize jIdx as 0
        int kIdx          = jIdx + 1 + reduced_idx;
        int pair_computes_this_row = neighborlistSize - 1; // initialize
        // looping over the triplets and storing the virial sum; have E3B evaluator take rij vectors, computed here
        */
        // declare the variables we need
        real3 pos_a2,pos_b2,pos_c2; // a1,b1,c1,a3,b3,c3 already declared
        real3 r_a1b2,r_a1c2,r_b1a2,r_c1a2;
        real3 r_a1b3,r_a1c3,r_b1a3,r_c1a3;
        real3 r_a2b3,r_a2c3,r_b2a3,r_c2a3;
        real  r_a1b2_scalar,r_a1c2_scalar,r_b1a2_scalar,r_c1a2_scalar;
        real  r_a1b3_scalar,r_a1c3_scalar,r_b1a3_scalar,r_c1a3_scalar;
        real  r_a2b3_scalar,r_a2c3_scalar,r_b2a3_scalar,r_c2a3_scalar;
       
        // In this loop, we will have the intermolecular OH vectors as i,j,k for a given triplet of molecules;
        // To get the force, we must interchange the indices; but, this would result in the same 
        // intermolecular OH vectors with re-labled indices;
        // instead, compute the distances once, shuffle the vectors in E3B's three body evaluator, 
        // and sum the forces on that side accordingly.
        /*
        while (pair_pair_idx < maxNumComputes) {
            // pair_pair_idx is up-to-date; reduced_idx was incremented, but has not been reduced
            // jIdx, reduced_idx might not be up-to-date;
            while (reduced_idx >= pair_computes_this_row) {
                // as long as pair_pair_idx < maxNumComputes, pair_computes_this_row is guaranteed 
                // to be larger than 0
                // ---- tested in python; for all values of pair_pair_idx [0...maxNumComputes),
                // gives unique pairs [j,k], k = [j+1....N-1], where N is neighborCount, and j is 0-indexed
                // increment jIdx
                jIdx++;
                // reduce the reduced_idx by pair_computes_this_row
                reduced_idx -= pair_computes_this_row;
                // the pair_computes_this_row decreases by 1
                pair_computes_this_row--;
                // our kIdx is now as (jIdx + 1) + reduced_idx;
                // --- reduced idx is now the number of advances /after/ jIdx + 1
                kIdx = jIdx + 1 + reduced_idx;
            }
            // we have our 'i' molecule in the i-j-k triplet;
            // pair_pair_idx specifies where in the j, k flattened array we are
            // --- we need to compute what j is, and then get k; both will be obtained from shared memory

            // load atom positions from shared memory
            pos_a2 = smem_pos[3*jIdx       + base_smem_idx];
            pos_b2 = smem_pos[3*jIdx + 1   + base_smem_idx];
            pos_c2 = smem_pos[3*jIdx + 2   + base_smem_idx];

            pos_a3 = smem_pos[3*kIdx       + base_smem_idx];
            pos_b3 = smem_pos[3*kIdx + 1   + base_smem_idx];
            pos_c3 = smem_pos[3*kIdx + 2   + base_smem_idx];

            // compute the 12 unique vectors for this triplet; pass these, and the force, and virials to evaluator
            // -- no possible way to put these in to shared memory; here, we must do redundant calculations.
            //    i.e., there are too many to put in to shared memory, and its probably faster to just 
            //    do them per-thread rather than try to communicate, esp. if threads are in lockstep and 
            //    doing same computation anyways;
            //    if they are not in lockstep, we would incur computational expense checking on the 
            //    off chance that two threads are computing the same i-j, same i-k (never would be both)
            // rij = ri - rj

            // i = 1, j = 2
            //
            // rij: i = a1, j = b2
            r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
            r_a1b2_scalar = length(r_a1b2);
            // rij: i = a1, j = c2
            r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
            r_a1c2_scalar = length(r_a1c2);
            // rij: i = b1, j = a2
            r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
            r_b1a2_scalar = length(r_b1a2);
            // rij: i = c1, j = a2
            r_c1a2 = bounds.minImage(pos_c1 - pos_a2);
            r_c1a2_scalar = length(r_c1a2); 
            // i = 1, j = 3
            //
            // rij: i = a1, j = b3
            r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
            r_a1b3_scalar = length(r_a1b3);
            // rij: i = a1, j = c3
            r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
            r_a1c3_scalar = length(r_a1c3);
            // rij: i = b1, j = a3
            r_b1a3 = bounds.minImage(pos_b1 - pos_a3);
            r_b1a3_scalar = length(r_b1a3);
            // rij: i = c1, j = a3
            r_c1a3 = bounds.minImage(pos_c1 - pos_a3);
            r_c1a3_scalar = length(r_c1a3);

            // i = 2, j = 3
            //
            // rij: i = a2, j = b3
            r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
            r_a2b3_scalar = length(r_a2b3);
            // rij: i = a2, j = c3
            r_a2c3 = bounds.minImage(pos_a2 - pos_c3);
            r_a2c3_scalar = length(r_a2c3);
            // rij: i = b2, j = a3
            r_b2a3 = bounds.minImage(pos_b2 - pos_a3);
            r_b2a3_scalar = length(r_b2a3);
            // rij: i = c2, j = a3
            r_c2a3 = bounds.minImage(pos_c2 - pos_a3);
            r_c2a3_scalar = length(r_c2a3);

            // send distance scalars and eng_sum scalars (by reference) to E3B energy evaluator
            eval.threeBodyEnergy(eng_sum_a, eng_sum_b, eng_sum_c,
                             r_a1b2_scalar, r_a1c2_scalar, r_b1a2_scalar, r_c1a2_scalar,
                             r_a1b3_scalar, r_a1c3_scalar, r_b1a3_scalar, r_c1a3_scalar,
                             r_a2b3_scalar, r_a2c3_scalar, r_b2a3_scalar, r_c2a3_scalar);


            pair_pair_idx += warpSize; // advance pair_pair_idx through the nlist by warpSize
            reduced_idx   += warpSize; // advanced reduced_idx as warpSize; will be reduced at next loop
        } // end while (pair_pair_idx < maxNumComputes)
        */
        // let's divide the j's into groups of 4..
        // initially, a thread starts on j = 0, 1, 2 or 3...
        int jWorkGroupSize = 4; // four threads to a j
        int idxInWarp = threadIdx.x % warpSize; // 0...31
        int thisThreadjIdxInit = (idxInWarp) / jWorkGroupSize; // 0...7
        int workGroupsPerWarp = warpSize / jWorkGroupSize;
        int idxInWorkGroup = idxInWarp % jWorkGroupSize; // 0,1,2,3 ...

        // ok, so we have 8 groups of threads (4 threads in a group) advancing over the j neighborlist;
        // they are incremented by 8 at the conclusion of the computation
        // --- the kIdxs are traversed by the 4 threads within a given workgroup, advanced by workGroupSize (4)
        for (int jIdx = thisThreadjIdxInit; jIdx < neighborlistSize; jIdx+=workGroupsPerWarp) {
            for (int kIdx = idxInWorkGroup; kIdx < neighborlistSize; kIdx+=jWorkGroupSize) {
                if (kIdx == jIdx) continue;
                // load atom positions from shared memory
                pos_a2 = smem_pos[3*jIdx       + base_smem_idx];
                pos_b2 = smem_pos[3*jIdx + 1   + base_smem_idx];
                pos_c2 = smem_pos[3*jIdx + 2   + base_smem_idx];

                pos_a3 = smem_pos[3*kIdx       + base_smem_idx];
                pos_b3 = smem_pos[3*kIdx + 1   + base_smem_idx];
                pos_c3 = smem_pos[3*kIdx + 2   + base_smem_idx];

                // rij = ri - rj

                // i = 1, j = 2
                //
                // rij: i = a1, j = b2
                r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
                r_a1b2_scalar = length(r_a1b2);
                // rij: i = a1, j = c2
                r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
                r_a1c2_scalar = length(r_a1c2);
                // rij: i = b1, j = a2
                r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
                r_b1a2_scalar = length(r_b1a2);
                // rij: i = c1, j = a2
                r_c1a2 = bounds.minImage(pos_c1 - pos_a2);
                r_c1a2_scalar = length(r_c1a2); 
                // i = 1, j = 3
                //
                // rij: i = a1, j = b3
                r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
                r_a1b3_scalar = length(r_a1b3);
                // rij: i = a1, j = c3
                r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
                r_a1c3_scalar = length(r_a1c3);
                // rij: i = b1, j = a3
                r_b1a3 = bounds.minImage(pos_b1 - pos_a3);
                r_b1a3_scalar = length(r_b1a3);
                // rij: i = c1, j = a3
                r_c1a3 = bounds.minImage(pos_c1 - pos_a3);
                r_c1a3_scalar = length(r_c1a3);

                // i = 2, j = 3
                //
                // rij: i = a2, j = b3
                r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
                r_a2b3_scalar = length(r_a2b3);
                // rij: i = a2, j = c3
                r_a2c3 = bounds.minImage(pos_a2 - pos_c3);
                r_a2c3_scalar = length(r_a2c3);
                // rij: i = b2, j = a3
                r_b2a3 = bounds.minImage(pos_b2 - pos_a3);
                r_b2a3_scalar = length(r_b2a3);
                // rij: i = c2, j = a3
                r_c2a3 = bounds.minImage(pos_c2 - pos_a3);
                r_c2a3_scalar = length(r_c2a3);

                // send distance scalars and eng_sum scalars (by reference) to E3B energy evaluator
                eval.threeBodyEnergy(eng_sum_a, eng_sum_b, eng_sum_c,
                                 r_a1b2_scalar, r_a1c2_scalar, r_b1a2_scalar, r_c1a2_scalar,
                                 r_a1b3_scalar, r_a1c3_scalar, r_b1a3_scalar, r_c1a3_scalar,
                                 r_a2b3_scalar, r_a2c3_scalar, r_b2a3_scalar, r_c2a3_scalar);
            }
        }
    } // end doing the pair-pair computes, getting positions from shared memory


    eng_sum_as_real3 = make_real3(eng_sum_a, eng_sum_b,eng_sum_c);
    
    __syncwarp(); // we need to wait until all threads are done computing their energies
    // now, do lane shifting to accumulate the energies in threadIdx.x % warpSize == 0

    // warpReduce all energies
    agg_eng_sum = warpReduceSum(eng_sum_as_real3,warpSize);
    // no syncing required after warp reductions
    
    // threadIdx.x % warpSize  == 0 does global write
    if (((threadIdx.x % warpSize) == 0) and (moleculeIdx < nMolecules)) {
        // load current energies on O, H, H of reference molecule
        real cur_eng_O  = perParticleEng[atomsReferenceMolecule.x];
        real cur_eng_H1 = perParticleEng[atomsReferenceMolecule.y];
        real cur_eng_H2 = perParticleEng[atomsReferenceMolecule.z];

        // add contributions from E3B; {.x, .y, .z} as O, H1, H2;
        // --- for some reason, doing this as simple 'real' instead of real3 caused compilation error
        cur_eng_O += agg_eng_sum.x;
        cur_eng_H1+= agg_eng_sum.y;
        cur_eng_H2+= agg_eng_sum.z;

        // write energies to global
        perParticleEng[atomsReferenceMolecule.x] = cur_eng_O;
        perParticleEng[atomsReferenceMolecule.y] = cur_eng_H1;
        perParticleEng[atomsReferenceMolecule.z] = cur_eng_H2;
    }

} // end compute_E3B_energy


// same thing, but now we compute the energy per molecule
// Compute e3b energy for triplet i-j-k, j on i's neighborlist, k not
// i.e., j is the bridging molecule
__global__ void compute_E3B_energy_edge
        (int nMolecules, 
         const int4 *__restrict__ atomsFromMolecule,         // by moleculeIdx, has atomIdxs
         const uint16_t *__restrict__ neighborCounts,        // gridGPULocal
         const uint *__restrict__ neighborlist,              // gridGPULocal
         const uint32_t * __restrict__ cumulSumMaxPerBlock,  // gridGPULocal
         int warpSize, 
         const real4 *__restrict__ xs,                       // as atomIdxs
         real * __restrict__ perParticleEng,                 // this is per-particle (O,H,H)
         BoundsGPU bounds, 
         int nMoleculesPerBlock,
         int maxNumNeighbors,
         EvaluatorE3B eval)
{

    // we have one molecule per warp, presumably;
    // store neighborlist as atomIdxsInMolecule
    extern __shared__ real3 smem_neighborAtomPos[];
    
    real3 *smem_pos = smem_neighborAtomPos;
    // my shared memory for this block will be as atom positions, by reference molecule... then jMoleculeIdxs, by reference molecule
    uint32_t *jMoleculeIdxs = (uint32_t*)&smem_pos[3*maxNumNeighbors*nMoleculesPerBlock]; 
    
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // get where our neighborlist for this molecule starts
    int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx,warpSize);

    real eng_sum_a, eng_sum_b, eng_sum_c;
    real3 agg_eng_sum;
    int4 atomsReferenceMolecule;
    real3 pos_a1, pos_b1, pos_c1;
    real3 pos_a2, pos_b2, pos_c2;
    int neighborlistSize,base_smem_idx,initNlistIdx,base_smem_idx_idxs;
    // this will be true or false for an entire warp
    if (moleculeIdx < nMolecules) {

        // this sum is only for molecule 1; therefore, drop 1 subscript; a,b,c denote O, H1, H2, respectively
        eng_sum_a = 0.0;
        eng_sum_b = 0.0;
        eng_sum_c = 0.0;
        /* NOTE to others: see the notation used in 
         * Kumar and Skinner, J. Phys. Chem. B., 2008, 112, 8311-8318
         * "Water Simulation Model with Explicit 3 Body Interactions"
         *
         * we use their notation for decomposing the molecules into constituent atoms a,b,c (oxygen, hydrogen, hydrogen)
         * and decomposing the given trimer into the set of molecules 1,2,3 (water molecule 1, 2, and 3)
         */
        atomsReferenceMolecule = atomsFromMolecule[moleculeIdx]; // get our reference molecule atom idxs
        
        // load referencePos of atoms from global memory;
        real4 pos_a1_whole = xs[atomsReferenceMolecule.x];
        real4 pos_b1_whole = xs[atomsReferenceMolecule.y];
        real4 pos_c1_whole = xs[atomsReferenceMolecule.z];

        // get positions as real3
        pos_a1 = make_real3(pos_a1_whole);
        pos_b1 = make_real3(pos_b1_whole);
        pos_c1 = make_real3(pos_c1_whole);
        
        neighborlistSize = neighborCounts[moleculeIdx];
        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        initNlistIdx= threadIdx.x % warpSize; // begins as 0...31
        int curNlistIdx = initNlistIdx;
        base_smem_idx = (threadIdx.x / warpSize) * maxNumNeighbors * 3; // this neighborlist begins at warpIdx * maxNumNeighbors * 3 (atoms perNeighbor)
        base_smem_idx_idxs = (threadIdx.x / warpSize) * (maxNumNeighbors+1);
        real4 xs_O_whole, xs_H1_whole, xs_H2_whole;
        /* LOAD NEIGHBOR ATOM POSITIONS IN TO SHARED MEMORY */
        // curNlistIdx goes as the set {0...31} + (32 * N), N = 0...ceil(numNeighbors/32)
        while (curNlistIdx < neighborlistSize) {
            // retrieve the neighborlist molecule idx corresponding to this thread
            int nlistIdx = baseIdx + initNlistIdx +  warpSize * (curNlistIdx/warpSize);
            int neighborMoleculeIdx = neighborlist[nlistIdx];                  // global memory access
            int4 neighborAtomIdxs    = atomsFromMolecule[neighborMoleculeIdx]; // global memory access
            
            // shared memory thus has a size of (3 * maxNumNeighbors * sizeof(real4)) -- we do not need M-site position for this potential
            int idx_in_smem_O = (3 * curNlistIdx) + base_smem_idx;                          // put H1, H2 directly after this O; base_smem_idx is by warpIdx
            xs_O_whole = xs[neighborAtomIdxs.x]; // global memory access
            xs_H1_whole= xs[neighborAtomIdxs.y]; // global memory access
            xs_H2_whole= xs[neighborAtomIdxs.z]; // global memory access
            
            pos_a2  = make_real3(xs_O_whole);
            pos_b2  = make_real3(xs_H1_whole);
            pos_c2  = make_real3(xs_H2_whole);
            
            smem_pos[idx_in_smem_O]   = pos_a2;
            smem_pos[idx_in_smem_O+1] = pos_b2;
            smem_pos[idx_in_smem_O+2] = pos_c2;
            
            // and also stored the molecule idx; start at base_smem_idx_idxs, advance as curNlistIdx
            jMoleculeIdxs[base_smem_idx_idxs + curNlistIdx] = neighborMoleculeIdx;
            curNlistIdx += warpSize;                                        // advance as a warpSize
        }

        if (threadIdx.x % warpSize == 0) {
            jMoleculeIdxs[base_smem_idx_idxs + neighborlistSize] = moleculeIdx;
        }
    }
    __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                  // --- this is cuda 9.0 function; only needed for Volta architectures (and later)
      
    real3 eng_sum_as_real3 = make_real3(0.0, 0.0, 0.0);

    // and now, do (i,j), k on j's nlist, k not on i's nlist 
    // ---- if any one of the rx1x3 distances are less than the 5.2, then k is on i's nlist;
    //      see FixE3B; the nlist cutoff is s.t.
    // ---- permuting the indices shouldn't matter for the energy, just the forces.
    if (moleculeIdx < nMolecules) {

        // ok, so, as a warp, we pick a j; then, a given threadIdx.x % warpSize gives its idx in j's nlist; while 
        // idx in j's nlist < j's neighbor counts, continue; then, advance j until we have computed all j's
        // --- for each j, skip (i,j,k) if k is on i's neighborlist (i.e., if
        real3 pos_a3,pos_b3,pos_c3;
        real3 r_a1b2,r_a1c2,r_b1a2,r_c1a2;
        real3 r_a1b3,r_a1c3,r_b1a3,r_c1a3;
        real3 r_a2b3,r_a2c3,r_b2a3,r_c2a3;
        real  r_a1b2_scalar,r_a1c2_scalar,r_b1a2_scalar,r_c1a2_scalar;
        real  r_a1b3_scalar,r_a1c3_scalar,r_b1a3_scalar,r_c1a3_scalar;
        real  r_a2b3_scalar,r_a2c3_scalar,r_b2a3_scalar,r_c2a3_scalar;
        real4 xs_O_whole, xs_H1_whole, xs_H2_whole;
        
        // LOOPING OVER MOLECULE 'i' NEIGHBORLIST TO GET J MOLECULE
        // get j index from global nlist, and load position from smem
        
        int workGroupSize   = 4;
        int idxInWarp       = threadIdx.x % warpSize;    // 0...31
        int numWorkGroups   = warpSize / workGroupSize;  // 8
        int jWorkGroup      = idxInWarp / workGroupSize; // 0...7
        int kIdxInWorkGroup = idxInWarp % workGroupSize; // 0,1,2,3
        

        for (int jIdx = jWorkGroup; jIdx < neighborlistSize; jIdx+=numWorkGroups) {

            // load j's positions; we'll definitely need these.
            // jIdx as 0... neighborlistSize provides easy access to i's neighborlist in smem.
            // --- these do not change except when j loop is indexed
            pos_a2 = smem_pos[3*jIdx     + base_smem_idx];
            pos_b2 = smem_pos[3*jIdx + 1 + base_smem_idx];
            pos_c2 = smem_pos[3*jIdx + 2 + base_smem_idx];

            // i = 2, j = 1
            // j-i-k, j is the bridging molecule
            // s.t. k is not on i's nlist


            // rij: i = a1, j = b2 ---> change this to label as r_
            r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
            r_a1b2_scalar = length(r_a1b2);
            // rij: i = a1, j = c2
            r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
            r_a1c2_scalar = length(r_a1c2);
            // rij: i = b1, j = a2
            r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
            r_b1a2_scalar = length(r_b1a2);
            // rij: i = c1, j = a2
            r_c1a2 = bounds.minImage(pos_c1 - pos_a2);
            r_c1a2_scalar = length(r_c1a2); 


            // we stored these in shared memory the first time we read them in
            int globalJIdx = jMoleculeIdxs[base_smem_idx_idxs + jIdx];
            // so, i need j's global molecule idx to get its neighborlist - global memory access
            // --- also, because each thread is accessing a constant j per 'for' loop, we do not want 
            //     the initNlistIdx here (next line copied from above; modified the line below it)
            
            // ok, all threads in a warp now have the globalJIdx; now, we iterate as a warp over j's neighborlist 
            // to get K, with checking that k not on i's neighborlist
            
            // actually, because of padding on neighborlists etc... and variability that might be incurred by 
            // modifying nlist size to permit fewer nlist computations etc., it is probably easiest to store
            // the idxs in a linear array in smem, and simply check that kIdx not in array.

            // getting j molecule's neighborcounts
            int jNeighCounts = neighborCounts[globalJIdx];

            // load where j molecule idx's neighborlist starts in global memory, using its global index
            int baseIdx_globalJIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, globalJIdx, warpSize);
           
            int k_idx_j_neighbors = 0; //TODO fix
            int initNlistIdx_jNlist = k_idx_j_neighbors;
            for (int k_idx_j_neighbors = kIdxInWorkGroup; k_idx_j_neighbors < jNeighCounts; k_idx_j_neighbors += workGroupSize) {
            //while (k_idx_j_neighbors < jNeighCounts) {
                // get k idx from the actual j neighborlist - first, advance nlistIdx
                bool compute_ijk = true; // default to true; if we find k on i's nlist, set to false; then, check boolean

                // see e.g. PairEvaluateIso
                int nlistIdx_jNlist = baseIdx_globalJIdx + initNlistIdx_jNlist + warpSize * (k_idx_j_neighbors/warpSize);
                // this is our k molecule idx
                int kIdx = neighborlist[nlistIdx_jNlist];                  // global memory access
                
                // check if k is on i's neighborlist, or is i itself; this amounts to checking ~ 30 integers
                // the good thing is: this should be broadcast to all threads, so no serialized access!
                for (int idxs_to_check=0; idxs_to_check < neighborlistSize+1; idxs_to_check++) {
                    int idx_on_i_nlist = jMoleculeIdxs[base_smem_idx_idxs + idxs_to_check];
                    if (kIdx == idx_on_i_nlist) compute_ijk = false;
                }

                if (compute_ijk) {
                    // load k positions from global memory and do the computation.
                    int4 neighborAtomIdxs    = atomsFromMolecule[kIdx]; // global memory access

                    xs_O_whole = xs[neighborAtomIdxs.x]; // global memory access
                    xs_H1_whole= xs[neighborAtomIdxs.y]; // global memory access
                    xs_H2_whole= xs[neighborAtomIdxs.z]; // global memory access
                    
                    pos_a3 = make_real3(xs_O_whole);
                    pos_b3 = make_real3(xs_H1_whole);
                    pos_c3 = make_real3(xs_H2_whole);

                    // i = 1, j = 3
                    //
                    // rij: i = a1, j = b3
                    r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
                    r_a1b3_scalar = length(r_a1b3);
                    // rij: i = a1, j = c3
                    r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
                    r_a1c3_scalar = length(r_a1c3);
                    // rij: i = b1, j = a3
                    r_b1a3 = bounds.minImage(pos_b1 - pos_a3);
                    r_b1a3_scalar = length(r_b1a3);
                    // rij: i = c1, j = a3
                    r_c1a3 = bounds.minImage(pos_c1 - pos_a3);
                    r_c1a3_scalar = length(r_c1a3);

                    // i = 2, j = 3
                    //
                    // rij: i = a2, j = b3
                    r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
                    r_a2b3_scalar = length(r_a2b3);
                    // rij: i = a2, j = c3
                    r_a2c3 = bounds.minImage(pos_a2 - pos_c3);
                    r_a2c3_scalar = length(r_a2c3);
                    // rij: i = b2, j = a3
                    r_b2a3 = bounds.minImage(pos_b2 - pos_a3);
                    r_b2a3_scalar = length(r_b2a3);
                    // rij: i = c2, j = a3
                    r_c2a3 = bounds.minImage(pos_c2 - pos_a3);
                    r_c2a3_scalar = length(r_c2a3);

                    // send distance scalars and eng_sum scalars (by reference) to E3B energy evaluator
                    eval.threeBodyEnergy(eng_sum_a, eng_sum_b, eng_sum_c,
                                     r_a1b2_scalar, r_a1c2_scalar, r_b1a2_scalar, r_c1a2_scalar,
                                     r_a1b3_scalar, r_a1c3_scalar, r_b1a3_scalar, r_c1a3_scalar,
                                     r_a2b3_scalar, r_a2c3_scalar, r_b2a3_scalar, r_c2a3_scalar);
                } // end if compute_ijk -- note that this is a significant source of thread divergence :(

                // finally, advance k_idx_j_neighbors as warpSize
                k_idx_j_neighbors += workGroupSize;
            } // end loop over j's neighborlist
        } // end loop over i's neighborlist

        // all interactions (i,j,k) were computed on other threads as (w.r.t. this one) (j,i,k) and (k,i,j)
        // therefore, we triple counted the interactions.  divide by three accordingly
        eng_sum_as_real3 = make_real3(eng_sum_a, eng_sum_b,eng_sum_c);
    }
    __syncwarp(); // we need to wait until all threads are done computing their energies
    // now, do lane shifting to accumulate the energies in threadIdx.x % warpSize == 0

    // warpReduce all energies
    agg_eng_sum = warpReduceSum(eng_sum_as_real3,warpSize);
    // no syncing required after warp reductions
    
    // threadIdx.x % warpSize  == 0 does global write
    if (((threadIdx.x % warpSize) == 0) and (moleculeIdx < nMolecules)) {
        // load current energies on O, H, H of reference molecule
        real cur_eng_O  = perParticleEng[atomsReferenceMolecule.x];
        real cur_eng_H1 = perParticleEng[atomsReferenceMolecule.y];
        real cur_eng_H2 = perParticleEng[atomsReferenceMolecule.z];

        // add contributions from E3B; {.x, .y, .z} as O, H1, H2;
        // --- for some reason, doing this as simple 'real' instead of real3 caused compilation error
        cur_eng_O += agg_eng_sum.x;
        cur_eng_H1+= agg_eng_sum.y;
        cur_eng_H2+= agg_eng_sum.z;

        // write energies to global
        perParticleEng[atomsReferenceMolecule.x] = cur_eng_O;
        perParticleEng[atomsReferenceMolecule.y] = cur_eng_H1;
        perParticleEng[atomsReferenceMolecule.z] = cur_eng_H2;
    }

} // end compute_E3B_energy

#endif /* __CUDACC__ */


