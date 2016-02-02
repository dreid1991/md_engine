#include "DataSet.h"
#include "State.h"
#include <boost/python.hpp>
void DataSetPython::process(int turn) {
    boost::python::call<void>(pyProcess, turn);
    /*
    PyObject_Print(res, stdout, Py_PRINT_RAW);
    //so calling through boost is a lot nicer since it'll tell you if the arguments are
//	PyObject *res = PyObject_CallFunction(pyProcess, (char *) "i", turn);
	num resC = 0;
    bool append = true;
	if (PyInt_Check(res)) {
		resC = (num) PyInt_AsLong(res);
        data.push_back(resC);
	} else if (PyFloat_Check(res)) {
		resC = (num) PyFloat_AsDouble(res);
        data.push_back(resC);
	} else if (res == Py_None) {
        append = false;
	} else {
        append = false;
    }
    if (append) {
        turns.push_back(state->turn);
    }
    */
}


void export_DataSetPython() {
    class_<DataSetPython, bases<DataSet> > ("DataSetPython")
        ;
}
