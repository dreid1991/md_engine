#pragma once
#ifndef FIXTHERMOSTAT_BASE_H
#define FIXTHERMOSTAT_BASE_H

#include <boost/python.hpp>
#include "BoundsGPU.h"
#include "cutils_math.h"
class FixThermostatBase {
protected:
    //ONE of these three groups will be used based on thermo type
    std::vector<double> intervals;
    std::vector<double> temps;

    boost::python::object tempFunc;

    double constTemp;
    
    int mode;

    int64_t turnBeginRun;
    int64_t turnFinishRun;
    int curIntervalIdx;

    bool finished; //for interval - don't repeat interval

    double currentTemp;

public:
    FixThermostatBase(boost::python::list intervalsPy, boost::python::list tempsPy);
    FixThermostatBase(boost::python::object tempFunc_);
    FixThermostatBase(double temp_);
    FixThermostatBase(){};
    void computeCurrentTemp(int64_t turn);
    double getCurrentTemp();
    void finishRun();
};
class SumVectorSqr3DOverWIf_Bounds {
public:
    float4 *fs;
    uint32_t groupTag;
    BoundsGPU bounds;
    SumVectorSqr3DOverWIf_Bounds(float4 *fs_, uint32_t groupTag_, BoundsGPU &bounds_) : fs(fs_), groupTag(groupTag_), bounds(bounds_) {}
    inline __host__ __device__ float process (float4 &velocity ) {
        return lengthSqrOverW(velocity);
    }
    inline __host__ __device__ float zero() {
        return 0;
    }
    inline __host__ __device__ bool willProcess(float4 *src, int idx) {
        float3 pos = make_float3(src[idx]);
        uint32_t atomGroupTag = * (uint32_t *) &(fs[idx].w);
        return (atomGroupTag & groupTag) && bounds.inBounds(pos);
    }
};

#endif
