#include "FixThermostatBase.h"
#include "Logging.h"
enum thermoType {interval, constant, pyFunc};
namespace py = boost::python;
FixThermostatBase::FixThermostatBase(py::list intervals_, py::list temps_) {
    mode = thermoType::interval;
    int len = boost::python::len(intervals_);
    for (int i=0; i<len; i++) {
        boost::python::extract<double> intPy(intervals_[i]);
        boost::python::extract<double> tempPy(temps_[i]);
        if (!intPy.check() or !tempPy.check()) {
            assert(intPy.check() and tempPy.check());
        }
        double interval = intPy;
        double temp = tempPy;
        intervals.push_back(interval);
        temps.push_back(temp);
    }
    curIntervalIdx = 0;
    finished = false;
    mdAssert(intervals[0] == 0 and intervals.back() == 1, "Invalid intervals given to thermostat");
}
FixThermostatBase::FixThermostatBase(double temp_) {
    mode = thermoType::constant;
    constTemp = temp_;
    mdAssert(constTemp > 0, "Invalid temperature given to thermostat");
}
FixThermostatBase::FixThermostatBase(py::object tempFunc_) {
    mode = thermoType::pyFunc;
    tempFunc = tempFunc_;
    mdAssert(PyCallable_Check(tempFunc.ptr()), "Must give callable function to thermostat");

}


void FixThermostatBase::computeCurrentTemp(int64_t turn) {
    if (mode == thermoType::interval) {
        if (finished) {
            currentTemp = temps.back();
        } else {
            double frac = (turn-turnBeginRun) / (double) (turnFinishRun - turnBeginRun);
            while (frac > intervals[curIntervalIdx+1] and curIntervalIdx < intervals.size()-1) {
                curIntervalIdx++;
            }
            double tempA = temps[curIntervalIdx];
            double tempB = temps[curIntervalIdx+1];
            double intA = intervals[curIntervalIdx];
            double intB = intervals[curIntervalIdx+1];
            double fracThroughInterval = (frac-intA) / (intB-intA);
            currentTemp = tempB*fracThroughInterval + tempA*(1-fracThroughInterval);
        }
    } else if (mode == thermoType::constant) {
        currentTemp = constTemp;
    } else if (mode == thermoType::pyFunc) {
        currentTemp = py::call<double>(tempFunc.ptr(), turnBeginRun, turnFinishRun, turn);
    }
}

double FixThermostatBase::getCurrentTemp() {
    return currentTemp;
}

void FixThermostatBase::finishRun() {
    finished = true;
}
