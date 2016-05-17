#pragma once
#ifndef FIX_CHARGEPAIR_DSF_H
#define FIX_CHARGEPAIR_DSF_H

//#include "AtomParams.h"
#include "GPUArrayTex.h"
#include "FixCharge.h"

class State;

void export_FixChargePairDSF();

class FixChargePairDSF : public FixCharge {

private:
    float A;  // temps for compute
    float shift;

protected:
    float alpha;
    float r_cut;

public:
    FixChargePairDSF(boost::shared_ptr<State> state_,
                     std::string handle_, std::string groupHandle_);

    void setParameters(float alpha_, float r_cut_);
    void compute(bool);

};

#endif
