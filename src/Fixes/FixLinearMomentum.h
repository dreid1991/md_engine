#pragma once
#ifndef FIXLINEARMOMENTUM_H
#define FIXLINEARMOMENTUM_H

#include "Fix.h"
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include "GPUArrayDeviceGlobal.h"
void export_FixLinearMomentum();
class FixLinearMomentum : public Fix {

private:
    GPUArrayDeviceGlobal<float4> sumMomentum;
    Vector dimensions;

    bool prepareForRun();
    void compute(bool);

public:
    FixLinearMomentum(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_,
                      int applyEvery = 1, Vector dimensions = Vector(1, 1, 1));

};

#endif
