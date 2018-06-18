#pragma once
#include "globalDefs.h"
#include "PairEvaluateIso.h"
#include <boost/shared_ptr.hpp>
#include "FixChargeEwald.h"
#include "FixChargePairDSF.h"
#include "ChargeEvaluatorNone.h"
class EvaluatorWrapper {
public:
    virtual void compute(int nAtoms, int nPerRingPoly, real4 *xs, 
                         real4 *fs, uint16_t *neighborCounts, uint *neighborlist, 
                         real4 *neighborlistPositions,
                         uint32_t *cumulSumMaxPerBlock, int warpSize, real *parameters, 
                         int numTypes,  BoundsGPU bounds, real onetwoStr, 
                         real onethreeStr, real onefourStr, Virial *virials, 
                         real *qs, real qCutoffSqr, int virialMode, 
                         int nThreadPerBlock, int nThreadPerAtom) {};


    virtual void energy(int nAtoms, int nPerRingPoly, real4 *xs, 
                        real *perParticleEng, uint16_t *neighborCounts, 
                        uint *neighborlist, uint32_t *cumulSumMaxPerBlock, 
                        int warpSize, real *parameters, int numTypes, 
                        BoundsGPU bounds, real onetwoStr, real onethreeStr, 
                        real onefourStr, real *qs, real qCutoffSqr, 
                        int nThreadPerBlock, int nThreadPerAtom) {};
    
    
    virtual void energyGroupGroup(int nAtoms, int nPerRingPoly, real4 *xs, 
                                  real4 *fs, real *perParticleEng, uint16_t *neighborCounts, 
                                  uint *neighborlist, uint32_t *cumulSumMaxPerBlock, 
                                  int warpSize, real *parameters, int numTypes, 
                                  BoundsGPU bounds, real onetwoStr, real onethreeStr, 
                                  real onefourStr, real *qs, real qCutoffSqr, 
                                  uint32_t tagA, uint32_t tagB, int nThreadPerBlock, 
                                  int nThreadPerAtom) {};
};





template <class PAIR_EVAL, bool COMP_PAIRS, int N_PARAM, class CHARGE_EVAL, bool COMP_CHARGES>
class EvaluatorWrapperImplement : public EvaluatorWrapper {
public:
    EvaluatorWrapperImplement(PAIR_EVAL pairEval_, CHARGE_EVAL chargeEval_) : pairEval(pairEval_), chargeEval(chargeEval_) {
    }
    PAIR_EVAL pairEval;
    CHARGE_EVAL chargeEval;
    virtual void compute(int nAtoms, int nPerRingPoly, real4 *xs, 
                         real4 *fs, uint16_t *neighborCounts, uint *neighborlist, 
                         real4 *neighborlistPositions,
                         uint32_t *cumulSumMaxPerBlock, int warpSize, real *parameters, 
                         int numTypes,  BoundsGPU bounds, real onetwoStr, 
                         real onethreeStr, real onefourStr, Virial *virials, 
                         real *qs, real qCutoff, int virialMode, 
                         int nThreadPerBlock, int nThreadPerAtom) 
    {
        if (COMP_PAIRS or COMP_CHARGES) {
            //printf("nAtons %d nTPB %d nTPA %d NBLOCK %d\n",  nAtoms, nThreadPerBlock, nThreadPerAtom, NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom));
            if (virialMode==2 or virialMode == 1) {
                if (nThreadPerAtom==1) {
                    compute_force_iso<PAIR_EVAL, COMP_PAIRS, 
                    N_PARAM, true, 
                    CHARGE_EVAL, COMP_CHARGES, 0> <<<NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom), 
                    nThreadPerBlock, N_PARAM*numTypes*numTypes*sizeof(real)>>>(nAtoms,nPerRingPoly, xs, 
                                                               fs, neighborCounts, neighborlist, neighborlistPositions, cumulSumMaxPerBlock, 
                                                               warpSize, parameters, numTypes, bounds, 
                                                               onetwoStr, onethreeStr, onefourStr, 
                                                               virials, qs, qCutoff*qCutoff, 
                                                               nThreadPerAtom, pairEval, chargeEval);
                } else {

                    compute_force_iso<PAIR_EVAL, COMP_PAIRS, 
                        N_PARAM, true, 
                        CHARGE_EVAL, COMP_CHARGES, 1> <<<NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom), 
                        nThreadPerBlock, 
                        N_PARAM*numTypes*numTypes*sizeof(real) + nThreadPerBlock*(sizeof(real3) + sizeof(Virial))>>>(nAtoms,
                                            nPerRingPoly, xs, 
                                            fs, neighborCounts, neighborlist, neighborlistPositions, cumulSumMaxPerBlock, 
                                            warpSize, parameters, numTypes, bounds, 
                                            onetwoStr, onethreeStr, onefourStr, 
                                            virials, qs, qCutoff*qCutoff, 
                                            nThreadPerAtom, pairEval, chargeEval);
                }
            } else {

                if (nThreadPerAtom==1) {
                    compute_force_iso<PAIR_EVAL, COMP_PAIRS, 
                        N_PARAM, false, 
                        CHARGE_EVAL, COMP_CHARGES, 0> <<<NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom), 
                        nThreadPerBlock, N_PARAM*numTypes*numTypes*sizeof(real)>>>(nAtoms,nPerRingPoly, xs, 
                                                            fs, neighborCounts, neighborlist, neighborlistPositions, cumulSumMaxPerBlock, 
                                                            warpSize, parameters, numTypes, bounds,
                                                            onetwoStr, onethreeStr, onefourStr, 
                                                            virials, qs, qCutoff*qCutoff,
                                                            nThreadPerAtom, pairEval, chargeEval);
                } else {
                    compute_force_iso<PAIR_EVAL, COMP_PAIRS,
                        N_PARAM, false, 
                        CHARGE_EVAL, COMP_CHARGES, 1> <<<NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom), 
                        nThreadPerBlock, N_PARAM*numTypes*numTypes*sizeof(real) + nThreadPerBlock*sizeof(real3)>>>(nAtoms,nPerRingPoly, xs,
                                                            fs, neighborCounts, neighborlist, neighborlistPositions, cumulSumMaxPerBlock,
                                                            warpSize, parameters, numTypes, bounds, 
                                                            onetwoStr, onethreeStr, onefourStr,
                                                            virials, qs, qCutoff*qCutoff, 
                                                            nThreadPerAtom, pairEval, chargeEval);
                }
            }
        }
    }
    
    virtual void energy(int nAtoms, int nPerRingPoly, real4 *xs, 
                        real *perParticleEng, uint16_t *neighborCounts, uint *neighborlist, 
                        uint32_t *cumulSumMaxPerBlock, int warpSize, real *parameters, 
                        int numTypes, BoundsGPU bounds, 
                        real onetwoStr, real onethreeStr, real onefourStr, 
                        real *qs, real qCutoff, int nThreadPerBlock, int nThreadPerAtom) {
        if (nThreadPerAtom==1) {
           compute_energy_iso<PAIR_EVAL, COMP_PAIRS, 
                    N_PARAM, CHARGE_EVAL, COMP_CHARGES, 0> <<<NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom), 
                    nThreadPerBlock, N_PARAM*numTypes*numTypes*sizeof(real)>>> (nAtoms, nPerRingPoly, xs, 
                                            perParticleEng, neighborCounts, neighborlist, 
                                            cumulSumMaxPerBlock, warpSize, parameters, 
                                            numTypes, bounds, 
                                            onetwoStr, onethreeStr, onefourStr, 
                                            qs, qCutoff*qCutoff, 
                                            nThreadPerAtom, pairEval, chargeEval);
        } else {
           compute_energy_iso<PAIR_EVAL, COMP_PAIRS, 
               N_PARAM, CHARGE_EVAL, COMP_CHARGES, 1> <<<NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom), 
                    nThreadPerBlock, N_PARAM*numTypes*numTypes*sizeof(real) + sizeof(real) * nThreadPerBlock>>> (nAtoms, nPerRingPoly, xs,
                            perParticleEng, neighborCounts, neighborlist, cumulSumMaxPerBlock, 
                            warpSize, parameters, numTypes, bounds, 
                            onetwoStr, onethreeStr, onefourStr, 
                            qs, qCutoff*qCutoff, 
                            nThreadPerAtom, pairEval, chargeEval);
        }
    }
    
    virtual void energyGroupGroup(int nAtoms, int nPerRingPoly, 
                                  real4 *xs, real4 *fs, real *perParticleEng, 
                                  uint16_t *neighborCounts, uint *neighborlist, 
                                  uint32_t *cumulSumMaxPerBlock, int warpSize, 
                                  real *parameters, int numTypes, BoundsGPU bounds, 
                                  real onetwoStr, real onethreeStr, real onefourStr, 
                                  real *qs, real qCutoff, 
                                  uint32_t tagA, uint32_t tagB, 
                                  int nThreadPerBlock, int nThreadPerAtom) {
        if (nThreadPerAtom==1) {
            compute_energy_iso_group_group<PAIR_EVAL, COMP_PAIRS, 
                N_PARAM, CHARGE_EVAL, COMP_CHARGES, 0> <<<NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom), 
                    nThreadPerBlock, N_PARAM*numTypes*numTypes*sizeof(real)>>> (nAtoms, nPerRingPoly, xs, 
                            fs, perParticleEng, neighborCounts, neighborlist, 
                            cumulSumMaxPerBlock, warpSize, parameters, 
                            numTypes, bounds, 
                            onetwoStr, onethreeStr, onefourStr, 
                            qs, qCutoff*qCutoff, 
                            tagA, tagB, nThreadPerAtom, 
                            pairEval, chargeEval);
        } else {
            compute_energy_iso_group_group<PAIR_EVAL, COMP_PAIRS, 
                N_PARAM, CHARGE_EVAL, COMP_CHARGES, 1> <<<NBLOCKTEAM(nAtoms, nThreadPerBlock, nThreadPerAtom), 
                    nThreadPerBlock, N_PARAM*numTypes*numTypes*sizeof(real) + sizeof(real) * nThreadPerBlock>>> (nAtoms, nPerRingPoly, xs, 
                            fs, perParticleEng, neighborCounts, neighborlist, 
                            cumulSumMaxPerBlock, warpSize, parameters, 
                            numTypes, bounds, 
                            onetwoStr, onethreeStr, onefourStr, 
                            qs, qCutoff*qCutoff, 
                            tagA, tagB, nThreadPerAtom, 
                            pairEval, chargeEval);
        }

    }

};

template<class PAIR_EVAL, int N_PARAM, bool COMP_PAIRS>
boost::shared_ptr<EvaluatorWrapper> pickEvaluator_CHARGE(PAIR_EVAL pairEval, Fix *chargeFix) {
    if (chargeFix == nullptr) {
        ChargeEvaluatorNone none;
        return boost::shared_ptr<EvaluatorWrapper> (dynamic_cast<EvaluatorWrapper *>( new EvaluatorWrapperImplement<PAIR_EVAL, COMP_PAIRS, N_PARAM, ChargeEvaluatorNone, false>(pairEval, none)));
    } else if (chargeFix->type == chargeEwaldType) {
        FixChargeEwald *f = dynamic_cast<FixChargeEwald *>(chargeFix);
        ChargeEvaluatorEwald chargeEval = f->generateEvaluator();
        return boost::shared_ptr<EvaluatorWrapper> (dynamic_cast<EvaluatorWrapper *>(new EvaluatorWrapperImplement<PAIR_EVAL, COMP_PAIRS, N_PARAM, ChargeEvaluatorEwald, true>(pairEval, chargeEval)));
    } else if (chargeFix->type == chargePairDSFType) {
        FixChargePairDSF *f = dynamic_cast<FixChargePairDSF *>(chargeFix);
        ChargeEvaluatorDSF chargeEval = f->generateEvaluator();
        return boost::shared_ptr<EvaluatorWrapper> (dynamic_cast<EvaluatorWrapper *>(new EvaluatorWrapperImplement<PAIR_EVAL, COMP_PAIRS, N_PARAM, ChargeEvaluatorDSF, true>(pairEval, chargeEval)));
    }
    // this will cause runtime failure anyways; so, just get it to compile.
    assert(false);
    // won't compile if only provided nullptr
    return boost::shared_ptr<EvaluatorWrapper>();
}

template <class PAIR_EVAL, int N_PARAM, bool COMP_PAIRS>
boost::shared_ptr<EvaluatorWrapper> pickEvaluator(PAIR_EVAL pairEval, Fix *chargeFix) {
    return pickEvaluator_CHARGE<PAIR_EVAL, N_PARAM, COMP_PAIRS>(pairEval, chargeFix);
}
