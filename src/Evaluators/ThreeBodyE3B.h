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
__global__ void compute_E3B_force
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

    int4 atomsReferenceMolecule;
    real3 pos_a1,pos_b1,pos_c1;
    int neighborlistSize, maxNumComputes,base_smem_idx,initNlistIdx;
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
        maxNumComputes = 0.5 * (neighborlistSize * (neighborlistSize - 1)); // number of unique triplets
        //if (threadIdx.x % 32 == 0) printf("moleculeIdx %d neighborlistSize %d\n",moleculeIdx,neighborlistSize);
        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        initNlistIdx= threadIdx.x % warpSize; // begins as 0...31
        int curNlistIdx = initNlistIdx;
        base_smem_idx = (threadIdx.x / warpSize) * maxNumNeighbors * 3; // this neighborlist begins at warpIdx * maxNumNeighbors * 3 (atoms perNeighbor)
        real3 xs_O, xs_H1, xs_H2;
        real4 xs_O_whole, xs_H1_whole, xs_H2_whole;
        real3 rij_a1a2;
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
            
            smem_neighborAtomPos[idx_in_smem_O]   = xs_O;
            smem_neighborAtomPos[idx_in_smem_O+1] = xs_H1;
            smem_neighborAtomPos[idx_in_smem_O+2] = xs_H2;
            
            rij_a1a2 = bounds.minImage(pos_a1 - xs_O);
            
            if (COMP_VIRIALS) {
                eval.twoBodyForce<true>(rij_a1a2,fs_sum_a,virialsSum_a);
            } else {
                eval.twoBodyForce<false>(rij_a1a2,fs_sum_a,virialsSum_a);
            }
            curNlistIdx += warpSize;                                        // advance as a warpSize
        }
        
    } // end (moleculeIdx < nMolecules)
    __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                  // --- this is cuda 9.0 function
    if (moleculeIdx < nMolecules) {    
        // ok, all neighboring atom positions are now stored sequentially in shared memory, grouped by neighboring molecule;
        int pair_pair_idx = initNlistIdx; // my index of the pair-pair computation, 0...31
        int reduced_idx   = pair_pair_idx;
        int jIdx          = 0;
        int kIdx          = jIdx + 1 + reduced_idx;
        int pair_computes_this_row = neighborlistSize - 1; // initialize
        // looping over the triplets and storing the virial sum; have E3B evaluator take rij vectors, computed here
        real3 pos_a2,pos_b2,pos_c2;
        real3 pos_a3,pos_b3,pos_c3;
        real3 r_a1b2,r_a1c2,r_b1a2,r_c1a2;
        real3 r_a1b3,r_a1c3,r_b1a3,r_c1a3;
        real3 r_a2b3,r_a2c3,r_b2a3,r_c2a3;

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
            pos_a2 = smem_neighborAtomPos[3*jIdx     + base_smem_idx];
            pos_b2 = smem_neighborAtomPos[3*jIdx + 1 + base_smem_idx];
            pos_c2 = smem_neighborAtomPos[3*jIdx + 2 + base_smem_idx];
            
            pos_a3 = smem_neighborAtomPos[3*kIdx     + base_smem_idx];
            pos_b3 = smem_neighborAtomPos[3*kIdx + 1 + base_smem_idx];
            pos_c3 = smem_neighborAtomPos[3*kIdx + 2 + base_smem_idx];

            // compute the 12 unique vectors for this triplet; pass these, and the force, and virials to evaluator
            // -- no possible way to put these in to shared memory; here, we must do redundant calculations.

            // rij = ri - rj

            // i = 1, j = 2
            //
            // rij: i = a1, j = b2
            r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
            // rij: i = a1, j = c2
            r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
            // rij: i = b1, j = a2
            r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
            // rij: i = c1, j = a2
            r_c1a2 = bounds.minImage(pos_c1 - pos_a2);
            
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
    }
    __syncwarp(); // we need to wait until all threads are done computing their forces (and virials)
    // now, do lane shifting to accumulate the forces (and virials, if necessary) in to threadIdx == 0 which does global write
    // warpReduce all forces; if virials, warpReduce all Virials as well
    // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
    warpReduceSum(fs_sum_a,warpSize);
    warpReduceSum(fs_sum_b,warpSize);
    warpReduceSum(fs_sum_c,warpSize);
    if (COMP_VIRIALS) {
        // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
        // virials for oxygen of reference molecule
        warpReduceSum(virialsSum_a,warpSize);
        // virials for hydrogen H1 of reference molecule
        warpReduceSum(virialsSum_b,warpSize);
        // virials for hydrogen H2 of reference molecule
        warpReduceSum(virialsSum_c,warpSize);
    
    }
    // no syncing required after warp reductions
    
    // threadIdx.x % warpSize does global write, iff. moleculeIdx < nMolecules
    if (((threadIdx.x % warpSize) == 0) and (moleculeIdx < nMolecules)) {
        // load curForce on O molecule
        real4 curForce_O = fs[atomsReferenceMolecule.x];
        real4 curForce_H1= fs[atomsReferenceMolecule.y];
        real4 curForce_H2= fs[atomsReferenceMolecule.z];

        // add contributions from E3B
        curForce_O += fs_sum_a;
        curForce_H1+= fs_sum_b;
        curForce_H2+= fs_sum_c;

        if (COMP_VIRIALS) {
            // load from global memory
            Virial virial_O = virials[atomsReferenceMolecule.x];
            Virial virial_H1= virials[atomsReferenceMolecule.y];
            Virial virial_H2= virials[atomsReferenceMolecule.z];
            // add contributions from E3B to global value
            virial_O += virialsSum_a;
            virial_H1+= virialsSum_b;
            virial_H2+= virialsSum_c;
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


   
// same thing, but now we compute the energy per molecule
__global__ void compute_E3B_energy
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
    
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // get where our neighborlist for this molecule starts
    int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx,warpSize);

    real eng_sum_a, eng_sum_b, eng_sum_c;
    int4 atomsReferenceMolecule;
    real3 pos_a1, pos_b1, pos_c1;
    real3 pos_a2, pos_b2, pos_c2;
    real3 pos_a3, pos_b3, pos_c3;
    real neighborlistSize,maxNumComputes;
    int base_smem_idx,initNlistIdx;
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
        maxNumComputes = 0.5 * (neighborlistSize * (neighborlistSize - 1)); // number of unique triplets
        //if (threadIdx.x % 32 == 0) printf("moleculeIdx %d neighborlistSize %d\n",moleculeIdx,neighborlistSize);
        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        initNlistIdx= threadIdx.x % warpSize; // begins as 0...31
        int curNlistIdx = initNlistIdx;
        base_smem_idx = (threadIdx.x / warpSize) * maxNumNeighbors * 3; // this neighborlist begins at warpIdx * maxNumNeighbors * 3 (atoms perNeighbor)
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
            
            smem_neighborAtomPos[idx_in_smem_O]   = pos_a2;
            smem_neighborAtomPos[idx_in_smem_O+1] = pos_b2;
            smem_neighborAtomPos[idx_in_smem_O+2] = pos_c2;
            
            real3 rij_a1a2 = bounds.minImage(pos_a1 - pos_a2);
            real rij_a1a2_scalar = length(rij_a1a2);
            eng_sum_a += 0.5 * eval.twoBodyEnergy(rij_a1a2_scalar); // 0.5 because we double count all two body pairs
            curNlistIdx += warpSize;                                        // advance as a warpSize
        }
    }
    __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                  // --- this is cuda 9.0 function
      
    real3 eng_sum_as_real3 = make_real3(0.0, 0.0, 0.0);
    if (moleculeIdx < nMolecules) {
        // ok, all neighboring atom positions are now stored sequentially in shared memory, grouped by neighboring molecule;
        int pair_pair_idx = initNlistIdx; // my index of the pair-pair computation, 0...31; i.e., threadIdx.x % warpSize
        int reduced_idx   = pair_pair_idx; // initialize reduced_idx as pair_pair_idx
        int jIdx          = 0; // initialize jIdx as 0
        int kIdx          = jIdx + 1 + reduced_idx;
        int pair_computes_this_row = neighborlistSize - 1; // initialize
        // looping over the triplets and storing the virial sum; have E3B evaluator take rij vectors, computed here

        // declare the variables we need
        real3 pos_a2,pos_b2,pos_c2; // a1,b1,c1,a3,b3,c3 already declared
        real3 r_a1b2,r_a1c2,r_b1a2,r_c1a2;
        real3 r_a1b3,r_a1c3,r_b1a3,r_c1a3;
        real3 r_a2b3,r_a2c3,r_b2a3,r_c2a3;
        real  r_a1b2_scalar,r_a1c2_scalar,r_b1a2_scalar,r_c1a2_scalar;
        real  r_a1b3_scalar,r_a1c3_scalar,r_b1a3_scalar,r_c1a3_scalar;
        real  r_a2b3_scalar,r_a2c3_scalar,r_b2a3_scalar,r_c2a3_scalar;
        
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

            // note that j and k are molecule neighbors indices in the neighborlist array; therefore, 
            // jpos_O = smem_neighborAtomPos[jIdx*3], H1, H2 are the next 2

            // we now have kIdx, jIdx;
            // let jIdx represent molecule 2, and kIdx represent molecule 3

            // load atom positions from shared memory
            pos_a2 = smem_neighborAtomPos[3*jIdx       + base_smem_idx];
            pos_b2 = smem_neighborAtomPos[3*jIdx + 1   + base_smem_idx];
            pos_c2 = smem_neighborAtomPos[3*jIdx + 2   + base_smem_idx];

            pos_a3 = smem_neighborAtomPos[3*kIdx       + base_smem_idx];
            pos_b3 = smem_neighborAtomPos[3*kIdx + 1   + base_smem_idx];
            pos_c3 = smem_neighborAtomPos[3*kIdx + 2   + base_smem_idx];

            // compute the 12 unique vectors for this triplet; pass these, and the force, and virials to evaluator
            // -- no possible way to put these in to shared memory; here, we must do redundant calculations.

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
        
        // divide all atom energies by 3.0 (we triple count each triplet)
        eng_sum_a /= 3.0;
        eng_sum_b /= 3.0;
        eng_sum_c /= 3.0;
        eng_sum_as_real3 = make_real3(eng_sum_a, eng_sum_b,eng_sum_c);
    } // end doing the pair-pair computes, getting positions from shared memory
    
    __syncwarp(); // we need to wait until all threads are done computing their energies
    // now, do lane shifting to accumulate the energies in threadIdx.x % warpSize == 0

    // warpReduce all energies
    warpReduceSum(eng_sum_as_real3,warpSize);
    // no syncing required after warp reductions
    
    // threadIdx.x % warpSize  == 0 does global write
    if (((threadIdx.x % warpSize) == 0) and (moleculeIdx < nMolecules)) {
        // load current energies on O, H, H of reference molecule
        real cur_eng_O  = perParticleEng[atomsReferenceMolecule.x];
        real cur_eng_H1 = perParticleEng[atomsReferenceMolecule.y];
        real cur_eng_H2 = perParticleEng[atomsReferenceMolecule.z];

        // add contributions from E3B; {.x, .y, .z} as O, H1, H2;
        // --- for some reason, doing this as simple 'real' instead of real3 caused compilation error
        cur_eng_O += eng_sum_as_real3.x;
        cur_eng_H1+= eng_sum_as_real3.y;
        cur_eng_H2+= eng_sum_as_real3.z;

        // write energies to global
        perParticleEng[atomsReferenceMolecule.x] = cur_eng_O;
        perParticleEng[atomsReferenceMolecule.y] = cur_eng_H1;
        perParticleEng[atomsReferenceMolecule.z] = cur_eng_H2;
    }

} // end compute_E3B_energy

#endif /* __CUDACC__ */


