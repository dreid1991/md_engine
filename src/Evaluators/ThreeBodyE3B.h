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
    extern __shared__ real4 smem_neighborAtomPos[];
    // smem_neighborIdxs should be of size (uint) * (nMoleculesPerBlock) * (maxNumNeighbors)
    // max shared memory per block
    // typically, since maxNumNeighbors pertains to molecules, this will be 

    // populate smem_neighborIdxs for this warp
    // blockDim.x gives number of threads in a block in x direction
    // gridDim.x gives number of blocks in a grid in the x direction
    // blockDim.x * gridDim.x gives n threads in grid in x direction
    // ---- blockDim.x * (nMoleculesPerBlock) + (threadIdx.x /32) gives moleculeIdx
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // get where our neighborlist for this molecule starts // TODO verify
    int baseIdx = baseNeighlistIdxFromIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx);
    
    // processing molecules by warp, i.e. a molecule's force computations are distributed across 32 threads in a threadblock;
    // then, 
    // the molecule idx will be given by blockIdx.x*blockDim.x + (threadIdx.x / warpSize)

    // this will be true or false for an entire warp
    if (moleculeIdx < nMolecules) {
        // aggregate virials and forces for the O, H, H of the reference molecules within this warp
        Virial virialsSum_a;
        Virial virialsSum_b;
        Virial virialsSum_c;
        if (COMP_VIRIALS) {
            virialsSum_a = Virial(0,0,0,0,0,0);
            virialsSum_b = Virial(0,0,0,0,0,0);
            virialsSum_c = Virial(0,0,0,0,0,0);
        }

        // this sum is only for molecule 1; therefore, drop 1 subscript; a,b,c denote O, H1, H2, respectively
        real3 fs_sum_a = make_real3(0.0, 0.0, 0.0);
        real3 fs_sum_b = make_real3(0.0, 0.0, 0.0);
        real3 fs_sum_c = make_real3(0.0, 0.0, 0.0);

        /* NOTE to others: see the notation used in 
         * Kumar and Skinner, J. Phys. Chem. B., 2008, 112, 8311-8318
         * "Water Simulation Model with Explicit 3 Body Interactions"
         *
         * we use their notation for decomposing the molecules into constituent atoms a,b,c (oxygen, hydrogen, hydrogen)
         * and decomposing the given trimer into the set of molecules 1,2,3 (water molecule 1, 2, and 3)
         */
        int4 atomsReferenceMolecule = atomsFromMolecule[moleculeIdx]; // get our reference molecule atom idxs
        
        // load referencePos of atoms from global memory;
        real4 pos_a1_whole = xs[atomsReferenceMolecule.x];
        real4 pos_b1_whole = xs[atomsReferenceMolecule.y];
        real4 pos_c1_whole = xs[atomsReferenceMolecule.z];

        // get positions as real3
        real3 pos_a1 = make_real3(pos_a1_whole);
        real3 pos_b1 = make_real3(pos_b1_whole);
        real3 pos_c1 = make_real3(pos_c1_whole);
        
        int neighborlistSize = neighborCounts[moleculeIdx];
        int maxNumComputes = 0.5 * (neighborlistSize * (neighborlistSize - 1)); // number of unique triplets

        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        int curNlistIdx = threadIdx.x % warpSize; // begins as 0...31
        int warpIdx = threadIdx.x / warpSize; // 0....8
        int base_smem_idx = warpIdx * maxNumNeighbors * 3; // this neighborlist begins at warpIdx * maxNumNeighbors * 3 (atoms perNeighbor)

        /* LOAD NEIGHBOR ATOM POSITIONS IN TO SHARED MEMORY */
        while (curNlistIdx < neighborlistSize) {
            // retrieve the neighborlist molecule idx corresponding to this thread
            uint neighborMoleculeIdx = neighborlist[curNlistIdx];           // global memory access
            int4 neighborAtomIdxs    = atomsFromMolecule[neighborMoleculeIdx]; // global memory access
            
            // shared memory thus has a size of (3 * maxNumNeighbors * sizeof(real4)) -- we do not need M-site position for this potential
            int idx_in_smem_O = (3 * curNlistIdx) + base_smem_idx;                          // put H1, H2 directly after this O

            smem_neighborAtomPos[idx_in_smem_O]   = xs[neighborAtomIdxs.x]; // global memory access
            smem_neighborAtomPos[idx_in_smem_O+1] = xs[neighborAtomIdxs.y]; // global memory access
            smem_neighborAtomPos[idx_in_smem_O+2] = xs[neighborAtomIdxs.z]; // global memory access

            // here we also compute the two body force
            real3 pos_a2 = make_real3(smem_neighborAtomPos[idx_in_smem_O]);
            real3 rij_a1a2 = bounds.minImage(pos_a1 - pos_a2);
            // this will yield warning 'no effect' if ThreeBodyE3B is not explicitly passed 
            // EvaluatorE3B; I suppose because the compiler assume pass by value 
            // if it can't see the templated evaluator?  So, for this evaluator, we don't use 
            // a template..
            if (COMP_VIRIALS) {
                eval.twoBodyForce<true>(rij_a1a2,fs_sum_a,virialsSum_a);
            } else {
                eval.twoBodyForce<false>(rij_a1a2,fs_sum_a,virialsSum_a);
            }
            curNlistIdx += warpSize;                                        // advance as a warpSize
        }
        __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                      // --- this is cuda 9.0 function
        
        // ok, all neighboring atom positions are now stored sequentially in shared memory, grouped by neighboring molecule;
        int pair_pair_idx = threadIdx.x % warpSize; // my index of the pair-pair computation, 0...31
        int reduced_idx   = pair_pair_idx;
        int jIdx          = 0;
        int kIdx          = jIdx + 1 + reduced_idx;
        int pair_computes_this_row = neighborlistSize - 1; // initialize
        // looping over the triplets and storing the virial sum; have E3B evaluator take rij vectors, computed here
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

            // note that j and k are molecule neighbors indices in the neighborlist array; therefore, 
            // jpos_O = smem_neighborAtomPos[jIdx*3], H1, H2 are the next 2

            // we now have kIdx, jIdx;

            // load atom positions from shared memory
            real4 pos_a2_whole = smem_neighborAtomPos[3*jIdx];
            real4 pos_b2_whole = smem_neighborAtomPos[3*jIdx + 1];
            real4 pos_c2_whole = smem_neighborAtomPos[3*jIdx + 2];

            real4 pos_a3_whole = smem_neighborAtomPos[3*kIdx];
            real4 pos_b3_whole = smem_neighborAtomPos[3*kIdx + 1];
            real4 pos_c3_whole = smem_neighborAtomPos[3*kIdx + 2];

            // cast molecule 2 positions as real3 for bounds.minImage;
            real3 pos_a2  = make_real3(pos_a2_whole);
            real3 pos_b2  = make_real3(pos_b2_whole);
            real3 pos_c2  = make_real3(pos_c2_whole);

            // cast k molecule positions as real3 for bounds.minImage;
            real3 pos_a3  = make_real3(pos_a3_whole);
            real3 pos_b3  = make_real3(pos_b3_whole);
            real3 pos_c3  = make_real3(pos_c3_whole);

            // compute the 12 unique vectors for this triplet; pass these, and the force, and virials to evaluator
            // -- no possible way to put these in to shared memory; here, we must do redundant calculations.

            // rij = ri - rj

            // i = 1, j = 2
            //
            // rij: i = a1, j = b2
            real3 r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
            // rij: i = a1, j = c2
            real3 r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
            // rij: i = b1, j = a2
            real3 r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
            // rij: i = c1, j = a2
            real3 r_c1a2 = bounds.minImage(pos_c1 - pos_a2);
            
            // i = 1, j = 3
            //
            // rij: i = a1, j = b3
            real3 r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
            // rij: i = a1, j = c3
            real3 r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
            // rij: i = b1, j = a3
            real3 r_b1a3 = bounds.minImage(pos_b1 - pos_a3);
            // rij: i = c1, j = a3
            real3 r_c1a3 = bounds.minImage(pos_c1 - pos_a3);

            // i = 2, j = 3
            //
            // rij: i = a2, j = b3
            real3 r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
            // rij: i = a2, j = c3
            real3 r_a2c3 = bounds.minImage(pos_a2 - pos_c3);
            // rij: i = b2, j = a3
            real3 r_b2a3 = bounds.minImage(pos_b2 - pos_a3);
            // rij: i = c2, j = a3
            real3 r_c2a3 = bounds.minImage(pos_c2 - pos_a3);

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
        
        __syncwarp(); // we need to wait until all threads are done computing their forces (and virials)
        // now, do lane shifting to accumulate the forces (and virials, if necessary) in to threadIdx == 0 which does global write

        // warpReduce all forces; if virials, warpReduce all Virials as well
        // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element
        warpReduceSum<real>(fs_sum_a.x,warpSize);
        warpReduceSum<real>(fs_sum_a.y,warpSize);
        warpReduceSum<real>(fs_sum_a.z,warpSize);

        warpReduceSum<real>(fs_sum_b.x,warpSize);
        warpReduceSum<real>(fs_sum_b.y,warpSize);
        warpReduceSum<real>(fs_sum_b.z,warpSize);

        warpReduceSum<real>(fs_sum_c.x,warpSize);
        warpReduceSum<real>(fs_sum_c.y,warpSize);
        warpReduceSum<real>(fs_sum_c.z,warpSize);

        if (COMP_VIRIALS) {
            // __shfl_down intrinsic only knows 32, 64 bit sizes; send element by element

            // virials for oxygen of reference molecule
            warpReduceSum<real>(virialsSum_a[0],warpSize);
            warpReduceSum<real>(virialsSum_a[1],warpSize);
            warpReduceSum<real>(virialsSum_a[2],warpSize);
            warpReduceSum<real>(virialsSum_a[3],warpSize);
            warpReduceSum<real>(virialsSum_a[4],warpSize);
            warpReduceSum<real>(virialsSum_a[5],warpSize);
            
            // virials for hydrogen H1 of reference molecule
            warpReduceSum<real>(virialsSum_b[0],warpSize);
            warpReduceSum<real>(virialsSum_b[1],warpSize);
            warpReduceSum<real>(virialsSum_b[2],warpSize);
            warpReduceSum<real>(virialsSum_b[3],warpSize);
            warpReduceSum<real>(virialsSum_b[4],warpSize);
            warpReduceSum<real>(virialsSum_b[5],warpSize);

            // virials for hydrogen H2 of reference molecule
            warpReduceSum<real>(virialsSum_c[0],warpSize);
            warpReduceSum<real>(virialsSum_c[1],warpSize);
            warpReduceSum<real>(virialsSum_c[2],warpSize);
            warpReduceSum<real>(virialsSum_c[3],warpSize);
            warpReduceSum<real>(virialsSum_c[4],warpSize);
            warpReduceSum<real>(virialsSum_c[5],warpSize);
        
        }
        
        // no syncing required after warp reductions
        
        // threadIdx.x % warpSize does global write
        if ((threadIdx.x % warpSize) == 0) {
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

    } // end (moleculeIdx < nMolecules)
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
         real * __restrict__ perParticleEng, // this is per-particle (O,H,H)
         BoundsGPU bounds, 
         int nMoleculesPerBlock,
         int maxNumNeighbors,
         EvaluatorE3B eval)
{

    // we have one molecule per warp, presumably;
    // store neighborlist as atomIdxsInMolecule
    extern __shared__ real4 smem_neighborAtomPos[];
    // smem_neighborIdxs should be of size (uint) * (nMoleculesPerBlock) * (maxNumNeighbors)
    // max shared memory per block
    // typically, since maxNumNeighbors pertains to molecules, this will be 

    // populate smem_neighborIdxs for this warp
    // blockDim.x gives number of threads in a block in x direction
    // gridDim.x gives number of blocks in a grid in the x direction
    // blockDim.x * gridDim.x gives n threads in grid in x direction
    // ---- blockDim.x * (nMoleculesPerBlock) + (threadIdx.x /32) gives moleculeIdx
    int moleculeIdx = blockIdx.x * nMoleculesPerBlock + (threadIdx.x / warpSize);
    
    // get where our neighborlist for this molecule starts
    // NOTE: this works because FixE3B passed our threadsPerBlock,threadsPerMolecule configuration 
    // to its grid when forming the neighborlist.
    int baseIdx = baseNeighlistIdxFromIndex(cumulSumMaxPerBlock, warpSize, moleculeIdx);
    
    // processing molecules by warp, i.e. a molecule's force computations are distributed across 32 threads in a threadblock;
    // then, 
    // the molecule idx will be given by blockIdx.x*blockDim.x + (threadIdx.x / warpSize)

    // this will be true or false for an entire warp
    if (moleculeIdx < nMolecules) {

        // this sum is only for molecule 1; therefore, drop 1 subscript; a,b,c denote O, H1, H2, respectively
        real  eng_sum_a = 0.0;
        real  eng_sum_b = 0.0;
        real  eng_sum_c = 0.0;
        
        /* NOTE to others: see the notation used in 
         * Kumar and Skinner, J. Phys. Chem. B., 2008, 112, 8311-8318
         * "Water Simulation Model with Explicit 3 Body Interactions"
         *
         * we use their notation for decomposing the molecules into constituent atoms a,b,c (oxygen, hydrogen, hydrogen)
         * and decomposing the given trimer into the set of molecules 1,2,3 (water molecule 1, 2, and 3)
         */
        int4 atomsReferenceMolecule = atomsFromMolecule[moleculeIdx]; // get our reference molecule atom idxs
        
        // load referencePos of atoms from global memory;
        real4 pos_a1_whole = xs[atomsReferenceMolecule.x];
        real4 pos_b1_whole = xs[atomsReferenceMolecule.y];
        real4 pos_c1_whole = xs[atomsReferenceMolecule.z];

        // get positions as real3
        real3 pos_a1 = make_real3(pos_a1_whole);
        real3 pos_b1 = make_real3(pos_b1_whole);
        real3 pos_c1 = make_real3(pos_c1_whole);
        
        int neighborlistSize = neighborCounts[moleculeIdx];
        int maxNumComputes = 0.5 * (neighborlistSize * (neighborlistSize - 1)); // number of unique triplets

        // put the neighbor positions in to shared memory, so that we don't have to consult global memory every time
        // -- here, since we also just traverse the neighborlist the one time, do the two body correction.
        int curNlistIdx = threadIdx.x % warpSize; // begins as 0...31
        int warpIdx = threadIdx.x / warpSize; // 0....8
        int base_smem_idx = warpIdx * maxNumNeighbors * 3; // this neighborlist begins at warpIdx * maxNumNeighbors * 3 (atoms perNeighbor)

        /* LOAD NEIGHBOR ATOM POSITIONS IN TO SHARED MEMORY */
        while (curNlistIdx < neighborlistSize) {
            // retrieve the neighborlist molecule idx corresponding to this thread
            uint neighborMoleculeIdx = neighborlist[curNlistIdx];           // global memory access
            int4 neighborAtomIdxs    = atomsFromMolecule[neighborMoleculeIdx]; // global memory access
            
            // shared memory thus has a size of (3 * maxNumNeighbors * sizeof(real4)) -- we do not need M-site position for this potential
            int idx_in_smem_O = (3 * curNlistIdx) + base_smem_idx;                          // put H1, H2 directly after this O

            smem_neighborAtomPos[idx_in_smem_O]   = xs[neighborAtomIdxs.x]; // global memory access
            smem_neighborAtomPos[idx_in_smem_O+1] = xs[neighborAtomIdxs.y]; // global memory access
            smem_neighborAtomPos[idx_in_smem_O+2] = xs[neighborAtomIdxs.z]; // global memory access

            // here we also compute the two body force
            real3 pos_a2 = make_real3(smem_neighborAtomPos[idx_in_smem_O]);
            real3 rij_a1a2 = bounds.minImage(pos_a1 - pos_a2);
            // this will yield warning 'no effect' if ThreeBodyE3B is not explicitly passed 
            // EvaluatorE3B; I suppose because the compiler assume pass by value 
            // if it can't see the templated evaluator?  So, for this evaluator, we don't use 
            // a template..
            real rij_a1a2_scalar = length(rij_a1a2);
            eng_sum_a += 0.5 * eval.twoBodyEnergy(rij_a1a2_scalar); // 0.5 because we double count all two body pairs
            curNlistIdx += warpSize;                                        // advance as a warpSize
        }
        __syncwarp(); // sync the warp; we now have all neighbor atoms for this molecule in shared memory
                      // --- this is cuda 9.0 function
        
        // ok, all neighboring atom positions are now stored sequentially in shared memory, grouped by neighboring molecule;
        int pair_pair_idx = threadIdx.x % warpSize; // my index of the pair-pair computation, 0...31
        int reduced_idx   = pair_pair_idx; // initialize reduced_idx as pair_pair_idx
        int jIdx          = 0; // initialize jIdx as 0
        int kIdx          = jIdx + 1 + reduced_idx;
        int pair_computes_this_row = neighborlistSize - 1; // initialize
        // looping over the triplets and storing the virial sum; have E3B evaluator take rij vectors, computed here
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
            real4 pos_a2_whole = smem_neighborAtomPos[3*jIdx];
            real4 pos_b2_whole = smem_neighborAtomPos[3*jIdx + 1];
            real4 pos_c2_whole = smem_neighborAtomPos[3*jIdx + 2];

            real4 pos_a3_whole = smem_neighborAtomPos[3*kIdx];
            real4 pos_b3_whole = smem_neighborAtomPos[3*kIdx + 1];
            real4 pos_c3_whole = smem_neighborAtomPos[3*kIdx + 2];

            // cast molecule 2 positions as real3 for bounds.minImage;
            real3 pos_a2  = make_real3(pos_a2_whole);
            real3 pos_b2  = make_real3(pos_b2_whole);
            real3 pos_c2  = make_real3(pos_c2_whole);

            // cast k molecule positions as real3 for bounds.minImage;
            real3 pos_a3  = make_real3(pos_a3_whole);
            real3 pos_b3  = make_real3(pos_b3_whole);
            real3 pos_c3  = make_real3(pos_c3_whole);

            // compute the 12 unique vectors for this triplet; pass these, and the force, and virials to evaluator
            // -- no possible way to put these in to shared memory; here, we must do redundant calculations.

            // rij = ri - rj

            // i = 1, j = 2
            //
            // rij: i = a1, j = b2
            real3 r_a1b2 = bounds.minImage(pos_a1 - pos_b2);
            real  r_a1b2_scalar = length(r_a1b2);
            // rij: i = a1, j = c2
            real3 r_a1c2 = bounds.minImage(pos_a1 - pos_c2);
            real  r_a1c2_scalar = length(r_a1c2);
            // rij: i = b1, j = a2
            real3 r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
            real  r_b1a2_scalar = length(r_b1a2);
            // rij: i = c1, j = a2
            real3 r_c1a2 = bounds.minImage(pos_c1 - pos_a2);
            real  r_c1a2_scalar = length(r_c1a2); 
            // i = 1, j = 3
            //
            // rij: i = a1, j = b3
            real3 r_a1b3 = bounds.minImage(pos_a1 - pos_b3);
            real  r_a1b3_scalar = length(r_a1b3);
            // rij: i = a1, j = c3
            real3 r_a1c3 = bounds.minImage(pos_a1 - pos_c3);
            real  r_a1c3_scalar = length(r_a1c3);
            // rij: i = b1, j = a3
            real3 r_b1a3 = bounds.minImage(pos_b1 - pos_a3);
            real  r_b1a3_scalar = length(r_b1a3);
            // rij: i = c1, j = a3
            real3 r_c1a3 = bounds.minImage(pos_c1 - pos_a3);
            real  r_c1a3_scalar = length(r_c1a3);

            // i = 2, j = 3
            //
            // rij: i = a2, j = b3
            real3 r_a2b3 = bounds.minImage(pos_a2 - pos_b3);
            real  r_a2b3_scalar = length(r_a2b3);
            // rij: i = a2, j = c3
            real3 r_a2c3 = bounds.minImage(pos_a2 - pos_c3);
            real  r_a2c3_scalar = length(r_a2c3);
            // rij: i = b2, j = a3
            real3 r_b2a3 = bounds.minImage(pos_b2 - pos_a3);
            real  r_b2a3_scalar = length(r_b2a3);
            // rij: i = c2, j = a3
            real3 r_c2a3 = bounds.minImage(pos_c2 - pos_a3);
            real  r_c2a3_scalar = length(r_c2a3);

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

        __syncwarp(); // we need to wait until all threads are done computing their forces (and virials)
        // now, do lane shifting to accumulate the forces (and virials, if necessary) in to threadIdx == 0 which does global write

        // warpReduce all energies
        warpReduceSum<real>(eng_sum_a,warpSize);
        warpReduceSum<real>(eng_sum_b,warpSize);
        warpReduceSum<real>(eng_sum_c,warpSize);
        
        // no syncing required after warp reductions
        
        // threadIdx.x % warpSize does global write
        if ((threadIdx.x % warpSize) == 0) {
            // load current energies on O, H, H of reference molecule
            real cur_eng_O  = perParticleEng[atomsReferenceMolecule.x];
            real cur_eng_H1 = perParticleEng[atomsReferenceMolecule.y];
            real cur_eng_H2 = perParticleEng[atomsReferenceMolecule.z];

            // add contributions from E3B
            cur_eng_O += eng_sum_a;
            cur_eng_H1+= eng_sum_b;
            cur_eng_H2+= eng_sum_c;

            // write energies to global
            perParticleEng[atomsReferenceMolecule.x] = cur_eng_O;
            perParticleEng[atomsReferenceMolecule.y] = cur_eng_H1;
            perParticleEng[atomsReferenceMolecule.z] = cur_eng_H2;
        }

    } // end (moleculeIdx < nMolecules)
} // end compute_E3B_energy

#endif /* __CUDACC__ */






