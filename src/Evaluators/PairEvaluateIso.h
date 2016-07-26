#include "BoundsGPU.h"
#include "cutils_func.h"
#include "Virial.h"
#include "helpers.h"
template <class T, int N, bool COMPUTEVIRIALS>
__global__ void compute_force_iso(int nAtoms, const float4 *__restrict__ xs, float4 *__restrict__ fs, const uint16_t *__restrict__ neighborCounts, const uint *__restrict__ neighborlist, const uint32_t * __restrict__ cumulSumMaxPerBlock, int warpSize, const float *__restrict__ parameters, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, Virial *__restrict__ virials, T eval) {
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
        Virial virialsSum = Virial(0, 0, 0, 0, 0, 0);
        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
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
                float params_pair[N];
                for (int pIdx=0; pIdx<N; pIdx++) {
                    params_pair[pIdx] = params_shr[pIdx][sqrIdx];
                }
                //evaluator.force(forceSum, dr, params_pair, lenSqr, multiplier);
                float rCutSqr = params_pair[0];
                if (lenSqr < rCutSqr) {
                    float3 force = eval.force(dr, params_pair, lenSqr, multiplier);
                    forceSum += force;
                    if (COMPUTEVIRIALS) {
                        computeVirial(virialsSum, force, dr);
                    }
                }
            }

        }   
    //    printf("LJ force %f %f %f \n", forceSum.x, forceSum.y, forceSum.z);
        float4 forceCur = fs[idx];
        forceCur += forceSum;
        fs[idx] = forceCur;
        if (COMPUTEVIRIALS) {
            virialsSum *= 0.5f;
            virials[idx] += virialsSum;
        }
    
        //fs[idx] += forceSum;

    }

}



template <class T, int N>
__global__ void compute_energy_iso(int nAtoms, float4 *xs, float *perParticleEng, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *parameters, int numTypes, BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, T evaluator) {
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
        float3 pos = make_float3(posWhole);

        float sumEng = 0;

        int numNeigh = neighborCounts[idx];
        //printf("start, end %d %d\n", start, end);
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
                    sumEng += evaluator.energy(params_pair, lenSqr, multiplier);
                }

            }

        }   
        //printf("force %f %f %f with %d atoms \n", forceSum.x, forceSum.y, forceSum.z, end-start);
        perParticleEng[idx] += sumEng;
        //fs[idx] += forceSum;

    }

}

