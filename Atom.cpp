
#include "Atom.h"
void export_Atom () { 
    class_<Atom>("Atom", init<int, int>())
        .def(init<Vector, int, int>())
        .def_readonly("id", &Atom::id)
        .def_readwrite("pos", &Atom::pos)
        .def_readwrite("vel", &Atom::vel)
        .def_readwrite("force", &Atom::force)
        .def_readwrite("forceLast", &Atom::forceLast)
        .def_readwrite("groupTag", &Atom::groupTag)
        .def_readonly("neighbors", &Atom::neighbors)
        .def_readwrite("mass", &Atom::mass)
        .def_readwrite("type", &Atom::type)
        .def("kinetic", &Atom::kinetic)
        ;

}

void export_Neighbor() {
    class_<Neighbor>("Neighbor", init<>())
        .def_readwrite("obj", &Neighbor::obj)
        .def_readwrite("offset", &Neighbor::offset)
        ;
}
