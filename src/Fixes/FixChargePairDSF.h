#pragma once
#ifndef FIX_CHARGEPAIR_DSF_H
#define FIX_CHARGEPAIR_DSF_H
//#include "AtomParams.h"
#include "GPUArrayTex.h"
#include "FixCharge.h"
class State;
using namespace std;
void export_FixChargePairDSF();

class FixChargePairDSF : public FixCharge {
  private:
	float A;//temps for compute
	float shift;
  protected:
	float alpha;
	float r_cut;
  public:
        void setParameters(float alpha_,float r_cut_);
        FixChargePairDSF(SHARED(State) state_, string handle_, string groupHandle_);

        void compute(bool);
	
};
#endif
