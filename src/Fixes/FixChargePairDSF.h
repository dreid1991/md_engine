#pragma once
#ifndef FIX_CHARGEPAIR_DSF_H
#define FIX_CHARGEPAIR_DSF_H

//#include "AtomParams.h"
#include "GPUArrayTex.h"
#include "FixCharge.h"
#include "ChargeEvaluatorDSF.h"

class State;

void export_FixChargePairDSF();

extern const std::string chargePairDSFType;
class FixChargePairDSF : public FixCharge {

private:
    real A;  // temps for compute
    real shift;

protected:
    real alpha;
    real r_cut;

public:
    FixChargePairDSF(boost::shared_ptr<State> state_,
                     std::string handle_, std::string groupHandle_);

    bool prepareForRun();
    void setParameters(real alpha_, real r_cut_);
    void compute(int);
    void singlePointEng(real *);
    void singlePointEngGroupGroup(real *, uint32_t, uint32_t);
    ChargeEvaluatorDSF generateEvaluator();
    void setEvalWrapper();
    std::vector<real> getRCuts();

};

#endif
