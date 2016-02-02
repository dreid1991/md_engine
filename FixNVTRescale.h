#ifndef FIXNVTRESCALE_H
#define FIXNVTRESCALE_H
#include "Fix.h"
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include "GPUArrayDevice.h"
void export_FixNVTRescale();
class FixNVTRescale : public Fix {
    vector<double> intervals;
    vector<double> temps;
    int curIdx;
    bool prepareForRun();
    void compute();
    bool downloadFromRun();
    GPUArrayDevice<float> tempGPU; //length one
    public:
        bool finished;
        FixNVTRescale(SHARED(State), string handle_, string groupHandle_, boost::python::list intervals, boost::python::list temps, int applyEvery=10);

};

#endif
