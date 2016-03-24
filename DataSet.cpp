#include "DataSet.h"
#include "State.h"

void DataSet::setCollectMode() {
    if (PyCallable_Check(collectGenerator)) {
        collectModeIsPython = true;
    } else {
        collectModeIsPython = false;
        cout << "Data set must either have python function or valid collect interval" << endl;
        assert(collectEvery > 0);
    }
}
void DataSet::prepareForRun() {
    setCollectMode();
}

void DataSet::takeCollectValues(int collectEvery_, PyObject *collectGenerator_) {
    collectEvery = collectEvery_;
    collectGenerator = collectGenerator;
    setCollectMode();
}

int64_t DataSet::getNextCollectTurn(int64_t turn) {
    if (collectModeIsPython) {
        int64_t userPythonFunctionResult = boost::python::call<int64_t>(collectGenerator, turn);
        assert(userPythonFunctionResult > turn);
        return userPythonFunctionResult;
    } else {
        return turn + collectEvery;
    }
}

void DataSet::setNextCollectTurn(int64_t turn) {
    nextCollectTurn = getNextCollectTurn(turn);
}
void export_DataSet() {
    class_<DataSet>("DataManager")
        .def_readonly("turns", &DataSet::turns)
        .def_readwrite("nextCollectTurn", &DataSet::nextCollectTurn)
 //       .def("getDataSet", &DataManager::getDataSet)
        ;
}
