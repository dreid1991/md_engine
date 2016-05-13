#include "Angle.h"
#include <boost/python.hpp>
AngleHarmonic::AngleHarmonic(Atom *a, Atom *b, Atom *c, double k_, double thetaEq_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    ids[2] = c->id;
    k = k_;
    thetaEq = thetaEq_;
    type = type_;
}

AngleHarmonic::AngleHarmonic(double k_, double thetaEq_, int type_) {
    for (int i=0; i<3; i++) {
        ids[i] = -1;
    }
    k = k_;
    thetaEq = thetaEq_;
    type = type_;
}


void AngleHarmonic::takeParameters(AngleHarmonic &other) {
    k = other.k;
    thetaEq = other.thetaEq;
}
void AngleHarmonic::takeIds(AngleHarmonic &other) {
    for (int i=0; i<3; i++) {
        ids[i] = other.ids[i];
    }

}

void AngleHarmonicGPU::takeParameters(AngleHarmonic &other) {
    k = other.k;
    thetaEq = other.thetaEq;
}
void AngleHarmonicGPU::takeIds(AngleHarmonic &other) {
    for (int i=0; i<3; i++) {
        ids[i] = other.ids[i];
    }
}


void export_AngleHarmonic() {
//need to expose ids or atoms somehow.  Could just do id1, 2, 3. Would prefer not to use any heap memory or pointers to make it trivially copyable  
    boost::python::class_<AngleHarmonic, SHARED(AngleHarmonic)> ( "AngleHarmonic", boost::python::init<>())
        .def_readwrite("thetaEq", &AngleHarmonic::thetaEq)
        .def_readwrite("k", &AngleHarmonic::k)
        .def_readwrite("type", &AngleHarmonic::type)
        .def_readonly("ids", &AngleHarmonic::ids)

    ;
}
