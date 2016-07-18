#pragma once
#ifndef DATASETUSER_H
#define DATASETUSER_H
#include "Python.h"
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <string.h>
class 
void export_DataSetUser();
namespace MD_ENGINE {
    enum COMPUTEMODE {INTERVAL, PYTHON};
    enum DATASCALARVECTOR {SCALAR, VECTOR};
    enum DATATYPE {TEMPERATURE, PRESSURE, ENERGY, BOUNDS};
    class DataSetUser {
    public:
        boost::python::list turns;
        boost::python::list vals;
        int dataType;
        int dataScalarVector;
        DataSetUser(int64_t, int, int, int);
        DataSetUser(int64_t, int, int, boost::python::object);
        int computeMode;
        int64_t nextCompute;


        boost::python::object pyFunc;
        PyObject *pyFuncRaw;
        int interval;
        void setNextTurn(int64_t currentTurn); //called externally 
    };
}



#endif
