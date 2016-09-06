#pragma once
#include "PairEvaluateIso.h"
#include <boost/shared_ptr.hpp>
#include "FixChargeEwald.h"
#include "FixChargePairDSF.h"
#include "ChargeEvaluatorNone.h"
class EvaluatorWrapper {
public:
    virtual void compute(int nAtoms, float4 *xs, float4 *fs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *parameters, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, Virial *virials, float *qs, float qCutoffSqr, bool computeVirials) {};
};

template <class PAIR_EVAL, int N_PARAM, class CHARGE_EVAL, bool COMP_CHARGES>
class EvaluatorWrapperImplement : public EvaluatorWrapper {
public:
    EvaluatorWrapperImplement(PAIR_EVAL pairEval_, CHARGE_EVAL chargeEval_) : pairEval(pairEval_), chargeEval(chargeEval_) {
    }
    PAIR_EVAL pairEval;
    CHARGE_EVAL chargeEval;
//<class PAIR_EVAL, int N_PARAM, bool COMP_VIRIALS, class CHARGE_EVAL, bool COMP_CHARGES>
    virtual void compute(int nAtoms, float4 *xs, float4 *fs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, int warpSize, float *parameters, int numTypes,  BoundsGPU bounds, float onetwoStr, float onethreeStr, float onefourStr, Virial *virials, float *qs, float qCutoffSqr, bool computeVirials) {
        //printf("forcers!\n");
        if (computeVirials) {
            compute_force_iso<PAIR_EVAL, N_PARAM, true, CHARGE_EVAL, COMP_CHARGES> <<<NBLOCK(nAtoms), PERBLOCK, N_PARAM*numTypes*numTypes*sizeof(float)>>>(nAtoms, xs, fs, neighborCounts, neighborlist, cumulSumMaxPerBlock, warpSize, parameters, numTypes, bounds, onetwoStr, onethreeStr, onefourStr, virials, qs, qCutoffSqr, pairEval, chargeEval);
        } else {
            compute_force_iso<PAIR_EVAL, N_PARAM, false, CHARGE_EVAL, COMP_CHARGES> <<<NBLOCK(nAtoms), PERBLOCK, N_PARAM*numTypes*numTypes*sizeof(float)>>>(nAtoms, xs, fs, neighborCounts, neighborlist, cumulSumMaxPerBlock, warpSize, parameters, numTypes, bounds, onetwoStr, onethreeStr, onefourStr, virials, qs, qCutoffSqr, pairEval, chargeEval);
        }
    }

};
template<class PAIR_EVAL, int N_PARAM>
boost::shared_ptr<EvaluatorWrapper> pickEvaluator_CHARGE(PAIR_EVAL pairEval, Fix *chargeFix) {
    if (chargeFix == nullptr) {
        ChargeEvaluatorNone none;
        return boost::shared_ptr<EvaluatorWrapper> (dynamic_cast<EvaluatorWrapper *>( new EvaluatorWrapperImplement<PAIR_EVAL, N_PARAM, ChargeEvaluatorNone, false>(pairEval, none)));
    } else if (chargeFix->type == chargeEwaldType) {
        FixChargeEwald *f = dynamic_cast<FixChargeEwald *>(chargeFix);
        ChargeEvaluatorEwald chargeEval = f->generateEvaluator();
        return boost::shared_ptr<EvaluatorWrapper> (dynamic_cast<EvaluatorWrapper *>(new EvaluatorWrapperImplement<PAIR_EVAL, N_PARAM, ChargeEvaluatorEwald, true>(pairEval, chargeEval)));
    } else if (chargeFix->type == chargePairDSFType) {
        FixChargePairDSF *f = dynamic_cast<FixChargePairDSF *>(chargeFix);
        ChargeEvaluatorDSF chargeEval = f->generateEvaluator();
        return boost::shared_ptr<EvaluatorWrapper> (dynamic_cast<EvaluatorWrapper *>(new EvaluatorWrapperImplement<PAIR_EVAL, N_PARAM, ChargeEvaluatorDSF, true>(pairEval, chargeEval)));
    }
    assert(false);
    return boost::shared_ptr<EvaluatorWrapper>(nullptr);
}

template <class PAIR_EVAL, int N_PARAM>
boost::shared_ptr<EvaluatorWrapper> pickEvaluator(PAIR_EVAL pairEval, Fix *chargeFix) {
    return pickEvaluator_CHARGE<PAIR_EVAL, N_PARAM>(pairEval, chargeFix);
}
