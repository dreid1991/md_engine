#pragma once
#include "BoundsGPU.h"
#include "cutils_func.h"
#include "Virial.h"
#include "helpers.h"
#include "SquareVector.h"
template <class PAIR_EVAL, bool COMP_PAIRS, int N_PARAM, bool COMP_VIRIALS, class CHARGE_EVAL, bool COMP_CHARGES>
__global__ void compute_force_iso
        (int nAtoms, 
	 int nPerRingPoly,
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
    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    //so we load in N_PARAM matrices which are of dimension numType*numTypes.  The matrices are arranged as linear blocks of data
    //paramsAll is the single big shared memory array that holds all of these parameters
    extern __shared__ float paramsAll[];
    int sqrSize = numTypes*numTypes;
    float *params_shr[N_PARAM];
    //then we take pointers into paramsAll.
    //
    //The order of the params_shr is given by the paramOrder array (see for example, FixLJCut.cu)
    if (COMP_PAIRS) {
        for (int i=0; i<N_PARAM; i++) {
            params_shr[i] = paramsAll + i * sqrSize;
        }
        //okay, so then we have a template to copy the global memory array parameters into paramsAll
        copyToShared<float>(parameters, paramsAll, N_PARAM*sqrSize);
        //then sync to let the threads finish their copying into shared memory
        __syncthreads();
    }

    // MW: NEED TO CHANGE ACCESS OF NEIGHBOR LIST BASED ON THREAD ID
    // This assumes that all ring polymers are the same size
    // this will change in later implementations where a variable number of beads may be used per RP
    int idx = GETIDX();
    if (idx < nAtoms) {
        Virial virialsSum = Virial(0, 0, 0, 0, 0, 0);
	// information based on ring polymer and bead
        int ringPolyIdx = idx / nPerRingPoly;	// which ring polymer
        int beadIdx     = idx % nPerRingPoly;	// which time slice

        //load where my neighborlist starts
        //int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx);
        float qi;

        //load charges if necessary
        if (COMP_CHARGES) {
            qi = qs[idx];
        }
        float4 posWhole = xs[idx];
        int type = __float_as_int(posWhole.w);
        float3 pos = make_float3(posWhole);

        float3 forceSum = make_float3(0, 0, 0);

        //how many neighbors do I have?
        //int numNeigh = neighborCounts[idx];
        int numNeigh = neighborCounts[ringPolyIdx];
        for (int i=0; i<numNeigh; i++) {
            //my neighbors, then, are spaced by warpSize
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            //The leftmost two bits in the neighbor entry say if it is a 1-2, 1-3, or 1-4 neighbor, or none of these
            uint neighDist = otherIdxRaw >> 30;
            float multiplier = multipliers[neighDist];
            //uint otherIdx = otherIdxRaw & EXCL_MASK;
            
            // Extract corresponding index for pair interaction (at same time slice)
            uint otherRPIdx = otherIdxRaw & EXCL_MASK;
	        uint otherIdx   = nPerRingPoly*otherRPIdx + beadIdx;  // atom = P*ring_polymer + k, k = 0,...,P-1
            float4 otherPosWhole = xs[otherIdx];

            //type is stored in w component of position
            int otherType = __float_as_int(otherPosWhole.w);
            float3 otherPos = make_float3(otherPosWhole);


            //based on the two atoms types, which index in each of the square matrices will I need to load from?
            int sqrIdx = squareVectorIndex(numTypes, type, otherType);
            float3 dr  = bounds.minImage(pos - otherPos);
            float lenSqr = lengthSqr(dr);
            //load that pair's parameters into a linear array to be send to the force evaluator
            float params_pair[N_PARAM];
            float rCutSqr;
            if (COMP_PAIRS) {
                for (int pIdx=0; pIdx<N_PARAM; pIdx++) {
                    params_pair[pIdx] = params_shr[pIdx][sqrIdx];
                }
                //we enforce that rCut is always the first parameter (for pairs at least, may need to be different for tersoff)
                rCutSqr = params_pair[0];
            }
            float3 force = make_float3(0, 0, 0);
            bool computedForce = false;
            if (COMP_PAIRS && lenSqr < rCutSqr) {
                //add to running total of the atom's forces
                force += pairEval.force(dr, params_pair, lenSqr, multiplier);
                computedForce = true;
            }
            if (COMP_CHARGES && lenSqr < qCutoffSqr) {
                //compute charge pair force if necessary
                float qj = qs[otherIdx];
                force += chargeEval.force(dr, lenSqr, qi, qj, multiplier);
                computedForce = true;
            }
            if (computedForce) {
                forceSum += force;
                if (COMP_VIRIALS) {
                    computeVirial(virialsSum, force, dr);
                }
            }

        }   
        float4 forceCur = fs[idx];
        forceCur += forceSum;
        //increment my forces by the total.  Note that each atom is calcu
        fs[idx] = forceCur;
        if (COMP_VIRIALS) {
            virialsSum *= 0.5f;
            virials[idx] += virialsSum;
        }
    

    }

}


//this is the analagous energy computation kernel for isotropic pair potentials.  See comments for force kernel, it's the same thing.

template <class PAIR_EVAL, bool COMP_PAIRS, int N, class CHARGE_EVAL, bool COMP_CHARGES>
__global__ void compute_energy_iso
        (int nAtoms, 
	 int nPerRingPoly,
         float4 *xs, 
         float *perParticleEng, 
         uint16_t *neighborCounts, 
         uint *neighborlist, 
         uint32_t *cumulSumMaxPerBlock, 
         int warpSize, 
         float *parameters, 
         int numTypes, 
         BoundsGPU bounds, 
         float onetwoStr, 
         float onethreeStr, 
         float onefourStr, 
         float *qs, 
         float qCutoffSqr, 
         PAIR_EVAL pairEval, 
         CHARGE_EVAL chargeEval) 
{
    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    extern __shared__ float paramsAll[];
    int sqrSize = numTypes*numTypes;
    float *params_shr[N];
    if (COMP_PAIRS) {
        for (int i=0; i<N; i++) {
            params_shr[i] = paramsAll + i * sqrSize;
        }
        copyToShared<float>(parameters, paramsAll, N*sqrSize);
        __syncthreads();    
    }

    // MW: NEED TO CHANGE ACCESS OF NEIGHBOR LIST BASED ON THREAD ID
    // This assumes that all ring polymers are the same size
    // this will change in later implementations where a variable number of beads may be used per RP
    int idx = GETIDX();
    if (idx < nAtoms) {
	// information based on ring polymer and bead
        int ringPolyIdx = idx / nPerRingPoly;	// which ring polymer
        int beadIdx     = idx % nPerRingPoly;	// which time slice

	//int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize,ringPolyIdx);
        float4 posWhole = xs[idx];
        //int type = * (int *) &posWhole.w;
        int type = __float_as_int(posWhole.w);
       // printf("type is %d\n", type);
        float qi;
        if (COMP_CHARGES) {
            qi = qs[idx];
        }
        float3 pos = make_float3(posWhole);

        float sumEng = 0;

        int numNeigh = neighborCounts[idx];
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            float multiplier = multipliers[neighDist];
            // Extract corresponding index for pair interaction (at same time slice)
            uint otherRPIdx = otherIdxRaw & EXCL_MASK;
	    uint otherIdx   = nPerRingPoly*otherRPIdx + beadIdx;  // atom = P*ring_polymer + k, k = 0,...,P-1
            //uint otherIdx = otherIdxRaw & EXCL_MASK;

            float4 otherPosWhole = xs[otherIdx];
            int otherType = __float_as_int(otherPosWhole.w);
            float3 otherPos = make_float3(otherPosWhole);
            float3 dr = bounds.minImage(pos - otherPos);
            float lenSqr = lengthSqr(dr);
            int sqrIdx = squareVectorIndex(numTypes, type, otherType);
            float rCutSqr;
            float params_pair[N];
            if (COMP_PAIRS) {
                for (int pIdx=0; pIdx<N; pIdx++) {
                    params_pair[pIdx] = params_shr[pIdx][sqrIdx];
                }
                rCutSqr = params_pair[0];
            }
            if (COMP_PAIRS && lenSqr < rCutSqr) {
                sumEng += pairEval.energy(params_pair, lenSqr, multiplier);
            }
            if (COMP_CHARGES && lenSqr < qCutoffSqr) {
                float qj = qs[otherIdx];
                float eng = chargeEval.energy(lenSqr, qi, qj, multiplier);
                //printf("len is %f\n", sqrtf(lenSqr));
                //printf("qi qj %f %f\n", qi, qj);
                //printf("eng is %f\n", eng);
                sumEng += eng;

            }


        }   
        perParticleEng[idx] += sumEng;

    }

}


//this is the group-group energy computation kernel for isotropic pair potentials.  



template <class PAIR_EVAL, bool COMP_PAIRS, int N, class CHARGE_EVAL, bool COMP_CHARGES>
__global__ void compute_energy_iso_group_group
        (int nAtoms, 
	 int nPerRingPoly,
         float4 *xs, 
         float4 *fs, 
         float *perParticleEng, 
         uint16_t *neighborCounts, 
         uint *neighborlist, 
         uint32_t *cumulSumMaxPerBlock, 
         int warpSize, 
         float *parameters, 
         int numTypes, 
         BoundsGPU bounds, 
         float onetwoStr, 
         float onethreeStr, 
         float onefourStr, 
         float *qs, 
         float qCutoffSqr, 
         uint32_t tagA,
         uint32_t tagB,
         PAIR_EVAL pairEval, 
         CHARGE_EVAL chargeEval) 
{
    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    extern __shared__ float paramsAll[];
    int sqrSize = numTypes*numTypes;
    float *params_shr[N];
    if (COMP_PAIRS) {
        for (int i=0; i<N; i++) {
            params_shr[i] = paramsAll + i * sqrSize;
        }
        copyToShared<float>(parameters, paramsAll, N*sqrSize);
        __syncthreads();    
    }

    // MW: NEED TO CHANGE ACCESS OF NEIGHBOR LIST BASED ON THREAD ID
    // This assumes that all ring polymers are the same size
    // this will change in later implementations where a variable number of beads may be used per RP
    int idx = GETIDX();
    if (idx < nAtoms) {
	// information based on ring polymer and bead
        int ringPolyIdx = idx / nPerRingPoly;	// which ring polymer
        int beadIdx     = idx % nPerRingPoly;	// which time slice

	//int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize,ringPolyIdx);
        uint32_t groupTagSelf = __float_as_uint(fs[idx].w);
        uint32_t groupTagCheck;
        if (groupTagSelf & tagA) {
            groupTagCheck = tagB;
        } else if (groupTagSelf & tagB) {
            groupTagCheck = tagA;
        } else {
            return;

        }
        float4 posWhole = xs[idx];
        //int type = * (int *) &posWhole.w;
        int type = __float_as_int(posWhole.w);
       // printf("type is %d\n", type);
        float qi;
        if (COMP_CHARGES) {
            qi = qs[idx];
        }
        float3 pos = make_float3(posWhole);

        float sumEng = 0;

        int numNeigh = neighborCounts[idx];
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            float multiplier = multipliers[neighDist];
            // Extract corresponding index for pair interaction (at same time slice)
            uint otherRPIdx = otherIdxRaw & EXCL_MASK;
	    uint otherIdx   = nPerRingPoly*otherRPIdx + beadIdx;  // atom = P*ring_polymer + k, k = 0,...,P-1
            //uint otherIdx = otherIdxRaw & EXCL_MASK;
            uint32_t otherGroupTag = __float_as_uint(fs[otherIdx].w);
            if (otherGroupTag & groupTagCheck) {

                float4 otherPosWhole = xs[otherIdx];
                int otherType = __float_as_int(otherPosWhole.w);
                float3 otherPos = make_float3(otherPosWhole);
                float3 dr = bounds.minImage(pos - otherPos);
                float lenSqr = lengthSqr(dr);
                int sqrIdx = squareVectorIndex(numTypes, type, otherType);
                float rCutSqr;
                float params_pair[N];
                if (COMP_PAIRS) {
                    for (int pIdx=0; pIdx<N; pIdx++) {
                        params_pair[pIdx] = params_shr[pIdx][sqrIdx];
                    }
                    rCutSqr = params_pair[0];
                }
                if (COMP_PAIRS && lenSqr < rCutSqr) {
                    sumEng += pairEval.energy(params_pair, lenSqr, multiplier);
                }
                if (COMP_CHARGES && lenSqr < qCutoffSqr) {
                    float qj = qs[otherIdx];
                    float eng = chargeEval.energy(lenSqr, qi, qj, multiplier);
                    //printf("len is %f\n", sqrtf(lenSqr));
                    //printf("qi qj %f %f\n", qi, qj);
                    //printf("eng is %f\n", eng);
                    sumEng += eng;

                }
            }


        }   
        perParticleEng[idx] += sumEng;

    }

}


/*
template <class PAIR_EVAL, bool COMP_PAIRS, int N, class CHARGE_EVAL, bool COMP_CHARGES>
__global__ void compute_energy_iso_group_group
        (int nAtoms, 
	 int nPerRingPoly,
         float4 *xs, 
         float4 *fs, 
         float *perParticleEng, 
         uint16_t *neighborCounts, 
         uint *neighborlist, 
         uint32_t *cumulSumMaxPerBlock, 
         int warpSize, 
         float *parameters, 
         int numTypes, 
         BoundsGPU bounds, 
         float onetwoStr, 
         float onethreeStr, 
         float onefourStr, 
         float *qs, 
         float qCutoffSqr, 
         uint32_t tagA,
         uint32_t tagB,
         PAIR_EVAL pairEval, 
         CHARGE_EVAL chargeEval) 
{
    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    extern __shared__ float paramsAll[];
    int sqrSize = numTypes*numTypes;
    float *params_shr[N];
    if (COMP_PAIRS) {
        for (int i=0; i<N; i++) {
            params_shr[i] = paramsAll + i * sqrSize;
        }
        copyToShared<float>(parameters, paramsAll, N*sqrSize);
        __syncthreads();    
    }

    // MW: NEED TO CHANGE ACCESS OF NEIGHBOR LIST BASED ON THREAD ID
    // This assumes that all ring polymers are the same size
    // this will change in later implementations where a variable number of beads may be used per RP
    int idx = GETIDX();
    if (idx < nAtoms) {
	// information based on ring polymer and bead
        int ringPolyIdx = idx / nPerRingPoly;	// which ring polymer
        int beadIdx     = idx % nPerRingPoly;	// which time slice

	//int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        int baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize,ringPolyIdx);
        uint32_t groupTagSelf = __float_as_uint(fs[idx].w);
        uint32_t groupTagCheck;
        if (groupTagSelf & tagA) {
            groupTagCheck = tagB;
        } else if (groupTagSelf & tagB) {
            groupTagCheck = tagA;
        } else {
            return;

        }
        float4 posWhole = xs[idx];
        //int type = * (int *) &posWhole.w;
        int type = __float_as_int(posWhole.w);
       // printf("type is %d\n", type);
        float qi;
        if (COMP_CHARGES) {
            qi = qs[idx];
        }
        float3 pos = make_float3(posWhole);

        float sumEng = 0;

        int numNeigh = neighborCounts[idx];
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            float multiplier = multipliers[neighDist];
            // Extract corresponding index for pair interaction (at same time slice)
            uint otherRPIdx = otherIdxRaw & EXCL_MASK;
	    uint otherIdx   = nPerRingPoly*otherRPIdx + beadIdx;  // atom = P*ring_polymer + k, k = 0,...,P-1
            //uint otherIdx = otherIdxRaw & EXCL_MASK;
            uint32_t otherGroupTag = __float_as_uint(fs[otherIdx].w);
            if (otherGroupTag & groupTagCheck) {

                float4 otherPosWhole = xs[otherIdx];
                int otherType = __float_as_int(otherPosWhole.w);
                float3 otherPos = make_float3(otherPosWhole);
                float3 dr = bounds.minImage(pos - otherPos);
                float lenSqr = lengthSqr(dr);
                int sqrIdx = squareVectorIndex(numTypes, type, otherType);
                float rCutSqr;
                float params_pair[N];
                if (COMP_PAIRS) {
                    for (int pIdx=0; pIdx<N; pIdx++) {
                        params_pair[pIdx] = params_shr[pIdx][sqrIdx];
                    }
                    rCutSqr = params_pair[0];
                }
                if (COMP_PAIRS && lenSqr < rCutSqr) {
                    sumEng += pairEval.energy(params_pair, lenSqr, multiplier);
                }
                if (COMP_CHARGES && lenSqr < qCutoffSqr) {
                    float qj = qs[otherIdx];
                    float eng = chargeEval.energy(lenSqr, qi, qj, multiplier);
                    //printf("len is %f\n", sqrtf(lenSqr));
                    //printf("qi qj %f %f\n", qi, qj);
                    //printf("eng is %f\n", eng);
                    sumEng += eng;

                }
            }


        }   
        perParticleEng[idx] += sumEng;

    }

}
*/
