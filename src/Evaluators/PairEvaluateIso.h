#pragma once
#ifndef PAIR_EVALUATOR_ISO_H
#define PAIR_EVALUATOR_ISO_H
#include "BoundsGPU.h"
#include "cutils_func.h"
#include "Virial.h"
#include "helpers.h"
#include "SquareVector.h"
template <class PAIR_EVAL, int N_PARAM, bool COMP_VIRIALS, class CHARGE_EVAL, bool COMP_CHARGES>
__global__ void compute_force_iso(int nAtoms, const float4 *__restrict__ xs, float4 *__restrict__ fs, const uint16_t *__restrict__ neighborCounts, const uint *__restrict__ neighborlist, const uint32_t * __restrict__ cumulSumMaxPerBlock, int warpSize, const float *__restrict__ parameters, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, Virial *__restrict__ virials, float *qs, float qCutoffSqr, PAIR_EVAL pairEval, CHARGE_EVAL chargeEval) {
    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    extern __shared__ float paramsAll[];
    int sqrSize = numTypes*numTypes;
    float *params_shr[N_PARAM];
    for (int i=0; i<N_PARAM; i++) {
        params_shr[i] = paramsAll + i * sqrSize;
    }
    copyToShared<float>(parameters, paramsAll, N_PARAM*sqrSize);
    __syncthreads();
    int idx = GETIDX();
    if (idx < nAtoms) {
        Virial virialsSum = Virial(0, 0, 0, 0, 0, 0);
        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        float qi;
        if (COMP_CHARGES) {
            qi = qs[idx];
        }
        float4 posWhole = xs[idx];
        int type = __float_as_int(posWhole.w);
        float3 pos = make_float3(posWhole);

        float3 forceSum = make_float3(0, 0, 0);

        int numNeigh = neighborCounts[idx];
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            float multiplier = multipliers[neighDist];
            if (multiplier) {
                uint otherIdx = otherIdxRaw & EXCL_MASK;
                float4 otherPosWhole = xs[otherIdx];
                int otherType = __float_as_int(otherPosWhole.w);
                float3 otherPos = make_float3(otherPosWhole);
                //then wrap and compute forces!
                int sqrIdx = squareVectorIndex(numTypes, type, otherType);
                float3 dr = bounds.minImage(pos - otherPos);
                float lenSqr = lengthSqr(dr);
                //printf("dr sqr pair is %f OMG UNCOMMENT IF MULT\n", lenSqr);
                float params_pair[N_PARAM];
                for (int pIdx=0; pIdx<N_PARAM; pIdx++) {
                    params_pair[pIdx] = params_shr[pIdx][sqrIdx];
                }
                //evaluator.force(forceSum, dr, params_pair, lenSqr, multiplier);
                float rCutSqr = params_pair[0];
                float3 force = make_float3(0, 0, 0);
                bool computedForce = false;
                if (lenSqr < rCutSqr) {
                    force += pairEval.force(dr, params_pair, lenSqr, multiplier);
                    computedForce = true;
                }
                if (COMP_CHARGES && lenSqr < qCutoffSqr) {
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

        }   
    //    printf("LJ force %f %f %f \n", forceSum.x, forceSum.y, forceSum.z);
        float4 forceCur = fs[idx];
        //float mult = 0.066 / 3.5;
        //printf("cur %f %f %f new %f %f %f\n", forceCur.x, forceCur.y, forceCur.z, mult*forceSum.x, mult*forceSum.y, mult*forceSum.z);
        forceCur += forceSum;
        fs[idx] = forceCur;
        if (COMP_VIRIALS) {
            virialsSum *= 0.5f;
            virials[idx] += virialsSum;
        }
    
        //fs[idx] += forceSum;

    }

}



template <class PAIR_EVAL, int N, class CHARGE_EVAL, bool COMP_CHARGES>
__global__ void compute_energy_iso(int nAtoms, float4 *xs, float *perParticleEng, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *parameters, int numTypes, BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, float *qs, float qCutoffSqr, PAIR_EVAL pairEval, CHARGE_EVAL chargeEval) {
    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    extern __shared__ float paramsAll[];
    int sqrSize = numTypes*numTypes;
    float *params_shr[N];
    for (int i=0; i<N; i++) {
        params_shr[i] = paramsAll + i * sqrSize;
    }
    copyToShared<float>(parameters, paramsAll, N*sqrSize);

    __syncthreads();    
    int idx = GETIDX();
    if (idx < nAtoms) {
        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
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
            if (multiplier) {
                uint otherIdx = otherIdxRaw & EXCL_MASK;
                float4 otherPosWhole = xs[otherIdx];
                int otherType = __float_as_int(otherPosWhole.w);
                float3 otherPos = make_float3(otherPosWhole);
                float3 dr = bounds.minImage(pos - otherPos);
                float lenSqr = lengthSqr(dr);
                int sqrIdx = squareVectorIndex(numTypes, type, otherType);
                float params_pair[N];
                for (int pIdx=0; pIdx<N; pIdx++) {
                    params_pair[pIdx] = params_shr[pIdx][sqrIdx];
                }
                float rCutSqr = params_pair[0];
                if (lenSqr < rCutSqr) {
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
#endif
