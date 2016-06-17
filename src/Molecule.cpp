#include "Molecule.h"
#include "boost_for_export.h"
namespace py = boost::python;
using namespace std;

#include "State.h"

Molecule::Molecule(State *state_, vector<int> &ids_) {
    state = state_;
    ids = ids_;
}

void Molecule::translate(Vector &v) {
    for (int id : ids) {
        Atom &a = state->idToAtom(id);
        a.pos += v;
    }
}
void Molecule::rotate(Vector &around, Vector &axis, double theta) {
    //also this 
}

Vector Molecule::COM() {
    //and this
}

void export_Molecule() {
    py::class_<Molecule> ("Molecule", py::no_init)
    .def_readonly("ids", &Molecule::ids)
    .def("translate", &Molecule::translate)
    ;
}
