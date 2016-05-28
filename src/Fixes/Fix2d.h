#pragma once
#ifndef FIX2D_H
#define FIX2D_H

#include "Fix.h"

class State;

const std::string _2dType = "2d";

void export_Fix2d();
class Fix2d : public Fix {

public:
    Fix2d(boost::shared_ptr<State> state_, std::string handle_, int applyEvery_)
      : Fix(state_, handle_, "all", _2dType, true, applyEvery_, false, false, 999)
    {   }

    void compute(bool);

};


#endif
