#pragma once
#ifndef FIXLINEARMOMENTUM_H

#include "Fix.h"
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include "GPUArrayDeviceGlobal.h"
void export_FixLinearMomentum();
class FixLinearMomentum : public Fix {
    bool prepareForRun();
    void compute(bool);
    GPUArrayDeviceGlobal<float4> sumMomentum;
    Vector dimensions;
    public:
        FixLinearMomentum(SHARED(State), string handle_, string groupHandle_, int applyEvery=1, Vector dimensions=Vector(1, 1, 1));

};

#endif
