#pragma once
#ifndef DATASETUSER_H
#define DATASETUSER_H
#include "Python.h"
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <string.h>
class State;
void export_DataSetUser();
namespace MD_ENGINE {
    enum COMPUTEMODE {INTERVAL, PYTHON};
    enum DATAMODE {SCALAR, TENSOR};
    enum DATATYPE {TEMPERATURE, PRESSURE, ENERGY, BOUNDS};
    class DataSetUser {
        State *state;
        void setRequiresFlags();
    public:
        boost::python::list turns;
        boost::python::list vals;
        int dataMode; 
        int dataType;
        uint32_t groupTag;
        DataSetUser(State *, boost::shared_ptr<DataComputer> computer_, uint32_t groupTag_, int, int, int);
        DataSetUser(State *, boost::shared_ptr<DataComputer> computer_, uint32_t groupTag_, int, int, boost::python::object);
        int computeMode;
        int64_t nextCompute;

        bool requiresVirials;
        bool requiresEnergy;

        void prepareForRun();
        void computeData();
        void appendData();

        boost::python::object pyFunc;
        PyObject *pyFuncRaw;
        int interval;
        void setNextTurn(int64_t currentTurn); //called externally 
        boost::shared_ptr<DataComputer> computer;
    };
}



#endif
