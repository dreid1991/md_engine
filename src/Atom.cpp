
#include "Atom.h"
#include "boost_for_export.h"
using namespace boost::python;


void Atom::setPos(Vector pos_) {
    isChanged = true;
    pos = pos_;
}
Vector Atom::getPos() {
    return pos;
}
void export_Atom () { 
    class_<Atom>("Atom", init<int, int>())
        .def(init<Vector, int, int>())
        .def_readonly("id", &Atom::id)
        .add_property("pos", &Atom::getPos, &Atom::setPos)
        //.def_readwrite("pos", &Atom::pos)
        .def_readwrite("vel", &Atom::vel)
        .def_readwrite("force", &Atom::force)
        .def_readwrite("groupTag", &Atom::groupTag)
        .def_readonly("neighbors", &Atom::neighbors)
        .def_readwrite("mass", &Atom::mass)
        .def_readwrite("q", &Atom::q)
        .def_readwrite("type", &Atom::type)
        .def("kinetic", &Atom::kinetic)
        .def_readwrite("isChanged", &Atom::isChanged)
        ;

}

void export_Neighbor() {
    class_<Neighbor>("Neighbor", init<>())
        .def_readwrite("obj", &Neighbor::obj)
        .def_readwrite("offset", &Neighbor::offset)
        ;
}
