#pragma once
#include "PairEvaluateIso.h"
class EvaluatorWrapper {
public:
    virtual void compute(int nAtoms, float4 *xs, float4 *fs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *parameters, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, Virial *virials) {};
};

template <class T, int N>
class EvaluatorWrapperImplement : public EvaluatorWrapper {
public:
    EvaluatorWrapperImplement(T e) {
        eval = e;
    }
    T eval;
    virtual void compute(int nAtoms, float4 *xs, float4 *fs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *parameters, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, Virial *virials) {
        compute_force_iso<T, N, false> <<<NBLOCK(nAtoms), PERBLOCK, N*numTypes*numTypes*sizeof(float)>>>(nAtoms, xs, fs, neighborCounts, neighborlist, cumulSumMaxPerBlock, warpSize, parameters, numTypes, bounds, onetwoStr, onethreeStr, onefourStr, virials, eval);

    }

};
