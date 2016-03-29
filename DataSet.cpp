#include "DataSet.h"
#include "State.h"
using namespace std;
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
void export_DataSet() {
    class_<DataSet, boost::noncopyable>("DataSet", no_init)
        .def_readonly("turns", &DataSet::turnsPy)
        .def_readonly("vals", &DataSet::valsPy)
        .def_readwrite("nextCollectTurn", &DataSet::nextCollectTurn)
 //       .def("getDataSet", &DataManager::getDataSet)
        ;
}
