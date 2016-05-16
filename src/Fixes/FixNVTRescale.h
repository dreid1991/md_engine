#pragma once
#ifndef FIXNVTRESCALE_H
#define FIXNVTRESCALE_H

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <string>
#include <vector>

#include "BoundsGPU.h"
#include "Fix.h"
#include "globalDefs.h"
#include "GPUArrayDeviceGlobal.h"

class Bounds;
class State;

void export_FixNVTRescale();
class FixNVTRescale : public Fix {
    vector<double> intervals;
    vector<double> temps;
    int curIdx;
    bool prepareForRun();
    void compute(bool);
    bool postRun();
    bool usingBounds;
    BoundsGPU boundsGPU;
    GPUArrayDeviceGlobal<float> tempGPU; //length two - first is temp, second is # atoms in group
    public:
        SHARED(Bounds) thermoBounds;
        bool finished;
        FixNVTRescale(SHARED(State), string handle_, string groupHandle_, boost::python::list intervals, boost::python::list temps, int applyEvery=10, SHARED(Bounds) thermoBounds_ = SHARED(Bounds)(NULL) );
        FixNVTRescale(SHARED(State), string handle_, string groupHandle_, vector<double> intervals, vector<double> temps, int applyEvery=10, SHARED(Bounds) thermoBounds_ = SHARED(Bounds)(NULL) );

};

#endif
