#include "DataSet.h"
#include "State.h"
namespace py = boost::python;
void DataSet::setCollectMode() {
    PyObject *func = collectGenerator.ptr();
    if (PyCallable_Check(func)) {
        collectModeIsPython = true;
    } else {
        collectModeIsPython = false;
        if (collectEvery <= 0) {
            cout << "Data set must either have python function or valid collect interval" << endl;
            assert(collectEvery > 0);
        }
    }
}

void DataSet::takeCollectValues(int collectEvery_, py::object collectGenerator_) {
    collectEvery = collectEvery_;
    collectGenerator = collectGenerator;
    setCollectMode();
}

int64_t DataSet::getNextCollectTurn(int64_t turn) {
    if (collectModeIsPython) {
        int64_t userPythonFunctionResult = boost::python::call<int64_t>(collectGenerator.ptr(), turn);
        assert(userPythonFunctionResult > turn);
        return userPythonFunctionResult;
    } else {
        return turn + collectEvery;
    }
}

void DataSet::setNextCollectTurn(int64_t turn) {
    nextCollectTurn = getNextCollectTurn(turn);
}
DataSet::DataSet(State *state_, uint32_t groupTag_, bool computingScalar_, bool computingVector_){
    state = state_;
    groupTag = groupTag_;
    requiresVirials = false;
    requiresEng = false;
    collectEvery = -1;
    turnScalarComputed = -1;
    turnVectorComputed = -1;
    lastScalarOnCPU = false;
    lastVectorOnCPU = false;
    computingScalar = computingScalar_;
    computingVector = computingVector_;

};
void export_DataSet() {
    boost::python::class_<DataSet,
                          boost::noncopyable>(
        "DataSet",
        boost::python::no_init
    )
    .def_readonly("turns", &DataSet::turnsPy)
    .def_readonly("vals", &DataSet::scalarsPy)
    .def_readonly("vectors", &DataSet::vectorsPy)
    .def_readwrite("nextCollectTurn", &DataSet::nextCollectTurn)
 //   .def("getDataSet", &DataManager::getDataSet)
    ;
}
