#include "Python.h"

#include "AtomParams.h"
#include "boost_for_export.h"
#include "State.h"
#define ARG_DEFAULT -1

namespace py=boost::python;

int AtomParams::addSpecies(std::string handle, double mass, double atomicNum) {
    //this is wrapped by state b/c fixes may need to update to accomodate more
    //atom types
    if (find(handles.begin(), handles.end(), handle) != handles.end()) {
        return -1;
    }
	handles.push_back(handle);
	int id = numTypes;
	numTypes ++;
	masses.push_back(mass);
    atomicNums.push_back(atomicNum);
	return id;
}


void AtomParams::clear() {
	handles = std::vector<std::string>();
	masses = std::vector<double>();
	numTypes = 0;
}

int AtomParams::typeFromHandle(const std::string &handle) const {
    auto it = find(handles.begin(), handles.end(), handle);
    if (it != handles.end()) {
        return it - handles.begin();
    }
    return -1;
}


void AtomParams::setValues(string handle, double mass, double atomicNum) {
    int idx = typeFromHandle(handle);
    if (mass != ARG_DEFAULT) {
        masses[idx] = mass;
    }
    if (atomicNum != ARG_DEFAULT) {
        atomicNums[idx] = atomicNum;
    }
}
void export_AtomParams() {
    py::class_<AtomParams >(
        "AtomParams"
    )
    .def("addSpecies", &AtomParams::addSpecies,
            (py::arg("handle"),
             py::arg("mass"),
             py::arg("atomicNum")=-1)
        )
    .def("typeFromHandle",  &AtomParams::typeFromHandle,
            (py::arg("handle"))
        )
    .def("setValues", &AtomParams::setValues,
            (py::arg("handle"),
             py::arg("mass")=ARG_DEFAULT,
             py::arg("atomicNum")=ARG_DEFAULT)
        )
    .def_readwrite("masses", &AtomParams::masses)
    .def_readonly("handles", &AtomParams::handles)//! \todo doesn't work
    .def_readonly("numTypes", &AtomParams::numTypes)
    ;
}
