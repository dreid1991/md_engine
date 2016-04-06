#pragma once
#ifndef FIX2D_H
#define FIX2D_H
class State;
#include "Fix.h"
void export_Fix2d();
class Fix2d : public Fix {
    public:
        Fix2d(SHARED(State) state_, string handle_, int applyEvery_) : Fix(state_, handle_, "all", _2dType, applyEvery_) {
            orderPreference = 999;
            forceSingle = true;
        };
        void compute(bool);

};


#endif
