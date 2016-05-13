#include "Bond.h"
#include "boost_for_export.h"
using namespace boost::python;

bool Bond::hasAtomId(int id) {
	return ids[0] == id or ids[1] == id;
}

int Bond::otherId(int id) const {
	if (ids[0] == id) {
		return ids[1];
	} else if (ids[1] == id) {
		return ids[0];
	} 
	return -1;
}

/*
void Bond::swap() {
    Atom *x = atoms[0];
    atoms[0] = atoms[1];
    atoms[1] = x;
}
*/


void export_BondHarmonic() {
  
    boost::python::class_<BondHarmonic,SHARED(BondHarmonic)> ( "BondHarmonic", boost::python::init<>())
//         .def(boost::python::init<int, int ,double, double,int>())
        .def_readonly("ids", &BondHarmonic::ids)
        .def_readwrite("k", &BondHarmonic::k)
        .def_readwrite("rEq", &BondHarmonic::rEq)
    ;
}
