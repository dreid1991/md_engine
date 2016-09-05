#pragma once
#include "PairEvaluateIso.h"
class EvaluatorWrapper {
public:
    virtual void compute(int nAtoms, float4 *xs, float4 *fs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *parameters, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, Virial *virials, float *qs, float qCutoffSqr) {};
};

template <class PAIR_EVAL, int N_PARAM, bool COMP_VIRIALS, class CHARGE_EVAL, bool COMP_CHARGES>
class EvaluatorWrapperImplement : public EvaluatorWrapper {
public:
    EvaluatorWrapperImplement(PAIR_EVAL pairEval_, CHARGE_EVAL chargeEval_) {
        pairEval = pairEval_;
        chargeEval = chargeEval_;
    }
    PAIR_EVAL pairEval;
    CHARGE_EVAL chargeEval;
//<class PAIR_EVAL, int N_PARAM, bool COMP_VIRIALS, class CHARGE_EVAL, bool COMP_CHARGES>
    virtual void compute(int nAtoms, float4 *xs, float4 *fs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *parameters, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, Virial *virials, float *qs, float qCutoffSqr) {
        //printf("forcers!\n");
        compute_force_iso<PAIR_EVAL, N_PARAM, COMP_VIRIALS, CHARGE_EVAL, COMP_CHARGES> <<<NBLOCK(nAtoms), PERBLOCK, N_PARAM*numTypes*numTypes*sizeof(float)>>>(nAtoms, xs, fs, neighborCounts, neighborlist, cumulSumMaxPerBlock, warpSize, parameters, numTypes, bounds, onetwoStr, onethreeStr, onefourStr, virials, qs, qCutoffSqr, pairEval, chargeEval);

    }

};


