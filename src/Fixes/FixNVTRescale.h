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

private:
    std::vector<double> intervals;
    std::vector<double> temps;
    int curIdx;

    bool usingBounds;
    BoundsGPU boundsGPU;
    GPUArrayDeviceGlobal<float> tempGPU;  // length two - first is temp, second is # atoms in group

    bool prepareForRun();
    void compute(bool);
    bool postRun();

public:
    boost::shared_ptr<Bounds> thermoBounds;
    bool finished;

    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_,
                  boost::python::list intervals, boost::python::list temps,
                  int applyEvery = 10, boost::shared_ptr<Bounds> thermoBounds_ = boost::shared_ptr<Bounds>(NULL));

    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_,
                  std::vector<double> intervals, std::vector<double> temps,
                  int applyEvery = 10, boost::shared_ptr<Bounds> thermoBounds_ = boost::shared_ptr<Bounds>(NULL));

};

#endif
