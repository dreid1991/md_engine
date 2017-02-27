#pragma once
#ifndef THREE_BODY_EVALUATE_ISO.H
#define THREE_BODY_EVALUATE_ISO.H

#include "BoundsGPU.h"
#include "cutils_func.h"
#include "Virial.h"
#include "helpers.h"
#include "SquareVector.h"

// consider: what needs to be templated here? //
__global__ void compute_three_body_iso 
        (int nAtoms, 
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



    // do some bookkeeping, as seen in PairEvaluateIso...
    //



    __syncthreads();
    int idx = GETIDX();

    if (idx < nAtoms) {
        
        

    }


}


// also, need to compute the potential energy (in addition to the forces)

#endif





