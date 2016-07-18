#include "DataSetUser.h"
#include "Logging.h"

using namespace MD_ENGINE;

DataSetUser::DataSetUser(int64_t currentTurn, int dataScalarVector_, int dataType_, boost::python::object pyFunc_) : computeMode(COMPUTEMODE::PYTHON), dataScalarVector(dataScalarVector_), dataType(dataType_), pyFunc(pyFunc_), pyFuncRaw(pyFunc_.ptr()) {
    mdAssert(PyCallable_Check(pyFuncRaw));
    setNextTurn(currentTurn);
}

DataSetUser::DataSetUser(int64_t currentTurn, int dataScalarVector_, int dataType_, int interval_) : computeMode(COMPUTEMODE::INTERVAL), dataScalarVector(dataScalarVector_), dataType(dataType_), interval(interval_) {
    nextCompute = currentTurn;

}

DataSetUser::setNextTurn(int64_t currentTurn) {
    if (computeMode == COMPUTEMODE::INTERVAL) {
        nextCompute = currentTurn + interval;
    } else {
        nextCompute = py::call<int64_t>(currentTurn);
    }
}

boost::python::object DataSetUser::getPyFunc() {
    return pyFunc;
}

void DataSetUser::setPyFunc(boost::python::object func_) {
    pyFunc = func_;
    pyFuncRaw = pyFunc.ptr();
    mdAssert(PyCallable_Check(pyFuncRaw));
}

void export_DataSetUser() {
    boost::python::class_<DataSeaUser, boost::noncopyable>("DataSet", boost::python::no_init)
    .def_readonly("turns", &DataSetUser::turns)
    .def_readonly("vals", &DataSetUser::vals)
    .def_readwrite("interval", &DataSet::interval)
    .add_property("pyFunc", &DataSetUser::getPyFunc, &DataSetUser::setPyFunc);
 //   .def("getDataSet", &DataManager::getDataSet)
    ;
}
