#include "Angle.h"
#include <boost/python.hpp>
AngleHarmonic::AngleHarmonic(Atom *a, Atom *b, Atom *c, double k_, double thetaEq_, int type_) {
    atoms[0] = a;
    atoms[1] = b;
    atoms[2] = c;
    k = k_;
    thetaEq = thetaEq_;
    type = type_;
}

AngleHarmonic::AngleHarmonic(double k_, double thetaEq_, int type_) {
    for (int i=0; i<3; i++) {
        atoms[i] = (Atom *) NULL;
    }
    k = k_;
    thetaEq = thetaEq_;
    type = type_;
}


void AngleHarmonic::takeValues(AngleHarmonic &angle) {
    k = angle.k;
    thetaEq = angle.thetaEq;
}

void AngleHarmonicGPU::takeIds(int *ids_) {
    ids[0] = ids_[0];
    ids[1] = ids_[1];
    ids[2] = ids_[2];
}
void AngleHarmonicGPU::takeValues(AngleHarmonic &other) {
    k = other.k;
    thetaEq = other.thetaEq;
}


void export_AngleHarmonic() {
//need to expose ids or atoms somehow.  Could just do id1, 2, 3. Would prefer not to use any heap memory or pointers to make it trivially copyable  
    boost::python::class_<AngleHarmonic, SHARED(AngleHarmonic)> ( "AngleHarmonic", boost::python::init<>())
        .def_readwrite("thetaEq", &AngleHarmonic::thetaEq)
        .def_readwrite("k", &AngleHarmonic::k)
        .def_readwrite("type", &AngleHarmonic::type)

    ;
}
