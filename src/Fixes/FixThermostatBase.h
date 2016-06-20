#pragma once
#ifndef FIXTHERMOSTAT_BASE_H
#define FIXTHERMOSTAT_BASE_H

#include <boost/python.hpp>
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


#endif
