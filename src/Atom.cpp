
#include "Atom.h"
#include "boost_for_export.h"
namespace py = boost::python;


void Atom::setPos(Vector &x) {
    isChanged = true;
    pos = x;
}
Vector Atom::getPos() {
    return pos;
}


void Atom::setVel(Vector &x) {
    isChanged = true;
    vel = x;
}
Vector Atom::getVel() {
    return vel;
}


void Atom::setForce(Vector &x) {
    isChanged = true;
    force = x;
}
Vector Atom::getForce() {
    return force;
}


std::string Atom::getType() {
    return handles->at(type);
}




void export_Atom () { 
    py::class_<Atom>("Atom", py::no_init)
        .def_readonly("id", &Atom::id)
        //.add_property("pos", &Atom::getPos, &Atom::setPos)
        .def_readwrite("pos", &Atom::pos)
        .def_readwrite("vel", &Atom::vel)
        .def_readwrite("force", &Atom::force)
        //.add_property("vel", &Atom::getVel, &Atom::setVel)
        //.add_property("force", &Atom::getForce, &Atom::setForce)
        .def_readwrite("groupTag", &Atom::groupTag)
        .def_readwrite("mass", &Atom::mass)
        .def_readwrite("q", &Atom::q)
        .add_property("type", &Atom::getType)
        .def("kinetic", &Atom::kinetic)
        .def_readwrite("isChanged", &Atom::isChanged)
        ;

}


