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
#include "FixThermostatBase.h"
class Bounds;
class State;

void export_FixNVTRescale();
class FixNVTRescale : public FixThermostatBase, public Fix {

private:
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
    FixNVTRescale() : Fix(boost::shared_ptr<State> (NULL), "a", "all", "st", false, false, false, 1){}; //remove this
    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, boost::python::list intervals, boost::python::list temps_, int applyEvery = 10, boost::shared_ptr<Bounds> thermoBounds_ = boost::shared_ptr<Bounds>(NULL));
    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, boost::python::object tempFunc_, int applyEvery = 10, boost::shared_ptr<Bounds> thermoBounds_ = boost::shared_ptr<Bounds>(NULL));
    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, double temp_, int applyEvery = 10, boost::shared_ptr<Bounds> thermoBounds_ = boost::shared_ptr<Bounds>(NULL));

    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_,
                  std::vector<double> intervals, std::vector<double> temps,
                  int applyEvery = 10, boost::shared_ptr<Bounds> thermoBounds_ = boost::shared_ptr<Bounds>(NULL));

};

#endif
