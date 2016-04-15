#include "Bond.h"
#include "boost_for_export.h"
using namespace boost::python;

bool Bond::hasAtom(Atom *a) {
	return atoms[0] == a or atoms[1] == a;
}

Atom *Bond::other(const Atom *a) const {
	if (atoms[0] == a) {
		return atoms[1];
	} else if (atoms[1] == a) {
		return atoms[0];
	} 
	return NULL;
}

Atom Bond::getAtom(int i) {
    return *atoms[i];
}

void Bond::swap() {
    Atom *x = atoms[0];
    atoms[0] = atoms[1];
    atoms[1] = x;
}


void export_BondHarmonic() {
  
    boost::python::class_<BondHarmonic,SHARED(BondHarmonic)> ( "BondHarmonic", boost::python::init<>())
//         .def(boost::python::init<int, int ,double, double,int>())
        .def_readwrite("id1", &BondHarmonic::id1)
        .def_readwrite("id2", &BondHarmonic::id2)
        .def_readwrite("k", &BondHarmonic::k)
        .def_readwrite("rEq", &BondHarmonic::rEq)
    ;
}
