#include "PythonOperation.h"
using namespace std;
using namespace boost::python;

PythonOperation::PythonOperation(string handle_, int operateEvery_, PyObject *operation_) {
    orderPreference = 0;//see header for comments
    operation = operation_;
    assert(PyCallable_Check(operation));
    operateEvery = operateEvery_;
    assert(operateEvery > 0);
    handle = handle_;
}

void PythonOperation::operate(int64_t turn) {
    boost::python::call<void>(operation, turn);
}

void export_PythonOperation() {
    class_<PythonOperation, SHARED(PythonOperation)> ("PythonOperation", init<string, int, PyObject*>(args("handle", "operateEvery", "operation")) )
        .def_readwrite("operateEvery", &PythonOperation::operateEvery)
        .def_readwrite("operation", &PythonOperation::operation)
        .def_readonly("handle", &PythonOperation::handle)
        ;
}
