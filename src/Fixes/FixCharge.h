#pragma once
#ifndef FIX_CHARGE_H
#define FIX_CHARGE_H
//#include "AtomParams.h"
#include "GPUArrayTex.h"
#include "Fix.h"
class State;
using namespace std;

void export_FixCharge();
class FixCharge : public Fix {
    public:
        FixCharge(SHARED(State) state_, string handle_, string groupHandle_,
                  string type_, bool forceSingle_);

        virtual void compute(bool){};
        bool prepareForRun();
	
};
#endif
