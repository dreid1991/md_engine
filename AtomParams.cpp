#include "AtomParams.h"
#include "State.h"

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

int AtomParams::typeFromHandle(std::string handle) {
    auto it = find(handles.begin(), handles.end(), handle);
    if (it != handles.end()) {
        return it - handles.begin();
    }
    return -1;
}

void export_AtomParams() {
    class_<AtomParams >("AtomParams")
        .def("addSpecies", &AtomParams::addSpecies, (python::arg("handle"), python::arg("mass"), python::arg("atomicNum")=-1) )
        .def_readwrite("masses", &AtomParams::masses)
        .def_readonly("handles", &AtomParams::handles)
        .def_readonly("numTypes", &AtomParams::numTypes)
        ;
}
