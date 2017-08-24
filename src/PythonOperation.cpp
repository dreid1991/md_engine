#include "PythonOperation.h"
#include "PythonHelpers.h"
using namespace std;
namespace py = boost::python;

PythonOperation::PythonOperation(string handle_, int operateEvery_, PyObject *operation_, bool synchronous_) {
    orderPreference = 0;//see header for comments
    operation = operation_;
    assert(PyCallable_Check(operation));
    operateEvery = operateEvery_;
    assert(operateEvery > 0);
    handle = handle_;
    synchronous = synchronous_;
}

void PythonOperation::operate(int64_t turn) {
	try {
		py::call<void>(operation, turn);
	} catch (boost::python::error_already_set &) {
		PythonHelpers::printErrors();
	}
}

void export_PythonOperation() {
	py::class_<PythonOperation, SHARED(PythonOperation)> ("PythonOperation", py::init<string, int, PyObject*>(py::args("handle", "operateEvery", "operation")) )
        .def_readwrite("operateEvery", &PythonOperation::operateEvery)
        .def_readwrite("operation", &PythonOperation::operation)
        .def_readonly("handle", &PythonOperation::handle)
        ;
}
