#include "DataSetUser.h"
#include "Logging.h"
#include "State.h"

using namespace MD_ENGINE;

DataSetUser::DataSetUser(State *state_, boost::shared_ptr<DataComputer> computer_, uint32_t groupTag_, int dataMode_, int dataType_, boost::python::object pyFunc_) : state(state_), computeMode(COMPUTEMODE::PYTHON), dataMode(dataMode_), dataType(dataType_), groupTag(groupTag_), pyFunc(pyFunc_), pyFuncRaw(pyFunc_.ptr()) {
    mdAssert(PyCallable_Check(pyFuncRaw));
    setNextTurn(state->turn);
    setRequiresFlags();
}

DataSetUser::DataSetUser(State *state, boost::shared_ptr<DataComputer> computer_, uint32_t groupTag_, int dataMode_, int dataType_, int interval_) : state(state_), computeMode(COMPUTEMODE::INTERVAL), dataMode(dataMode_), dataType(dataType_), groupTag(groupTag_), interval(interval_) {
    nextCompute = state->turn;
    setRequiresFlags();

}
void DataSetUser::prepareForRun() {
    compute->computingScalar = false;
    compute->computingVector = false;
    if (dataMode == DATAMODE::SCALAR) {
        compute->computingScalar = true;
    } else if (dataMode == DATAMODE::TENSOR) {
        compute->computingTensor = true;
    }
    compute->prepareForRun();
    
}
void DataSetUser::computeData() {
    if (dataMode == DATAMODE::SCALAR) {
        compute->computeScalar_GPU(true, groupTag);
    } else if (dataMode == DATAMODE::TENSOR) {
        compute->computeTensor_GPU(true, groupTag);
    }
    turns.append(state->turn);
}

void DataSetUser::appendData() {
    if (dataMode == DATAMODE::SCALAR) {
        compute->computeScalar_CPU();
        compute->appendScalar(vals);
    } else if (dataMode == DATAMODE::TENSOR) {
        compute->computeTensor_CPU();
        compute->appendTensor(vals);
    }
}

void DataSetUser::setRequiresFlags() {
    requiresVirials = false;
    requiresEnergy = false;
    if (dataType == DATATYPE::PRESSURE) {
        requiresVirials = true;
    }
    if (dataType == DATATYPE::ENERGY) {
        requiresEnergy = true;
    }
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
