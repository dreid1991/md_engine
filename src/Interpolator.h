#pragma once
#ifndef FIXINTERPOLATOR_BASE_H
#define FIXINTERPOLATOR_BASE_H

#include <boost/python.hpp>
#include "BoundsGPU.h"
#include "cutils_math.h"
class Interpolator {
public:
    //ONE of these three groups will be used based on thermo type
    std::vector<double> intervals;
    std::vector<double> vals;

    boost::python::object valFunc;

    double constVal;
    
    int mode;

    int64_t turnBeginRun;
    int64_t turnFinishRun;
    int curIntervalIdx;

    bool finished; //for interval - don't repeat interval

    double currentVal;

    Interpolator(boost::python::list intervalsPy, boost::python::list valsPy);
    Interpolator(boost::python::object valFunc_);
    Interpolator(double val_);
    Interpolator(){};
    void computeCurrentVal(int64_t turn);
    double getCurrentVal();
    void finishRun();
};


#endif
