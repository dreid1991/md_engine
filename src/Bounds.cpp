
#include <string>
#include <iostream>

#include "Bounds.h"
#include "State.h"
#include "AtomParams.h"

using namespace std;

void Bounds::handle2d() {
    if (state->is2d) {
        lo[2] = -0.5;
        hi[2] = +0.5;
        trace[2] = 1.0;
        sides[2] = Vector(0, 0, 1.0);
    }
}

BoundsGPU Bounds::makeGPU() {
    float3 sidesGPU[3];
    for (int i=0; i<3; ++i) {
        sidesGPU[i] = sides[i].asFloat3();
    }
    bool *periodic = state->periodic;
    return BoundsGPU(lo.asFloat3(), sidesGPU, make_float3((int)periodic[0],
                                                          (int)periodic[1],
                                                          (int)periodic[2])
    );
}

bool Bounds::atomInBounds(Atom &a) {
    for (int i=0; i<3; i++) {
        if (not (a.pos[i] >= lo[i] and a.pos[i] <= hi[i])) {
            return false;
        }
    }
    return true;
}

double Bounds::volume() {
    double v = 1;
    for (int i=0; i<3; i++) {
        v *= sides[i][i];
    }
    return v;
}


bool Bounds::skew(Vector skewBy) {
    skewBy[2] = 0;
    sides[0][1] += skewBy[1];
    sides[1][0] += skewBy[0];
    trace += skewBy;
    return true;
}

double Bounds::getSkewX() {
    return atan2(sides[0][1], sides[0][0]);
}

double Bounds::getSkewY() {
    return atan2(sides[1][0], sides[1][1]);
}

double Bounds::getSkew(int idx) {
    if (idx == 0) {
        return getSkewX();
    }   // return y skew if not x
    return getSkewY();
}

Bounds Bounds::unskewed() {
    Vector hiNew = lo;
    for (int i=0; i<3; i++) {
        hiNew[i] = lo[i] + sides[i][i];
    }
    return Bounds(state, lo, hiNew);
}

bool Bounds::isSkewed() {
    for (int i=0; i<3; i++) {
        Vector test(1, 1, 1);
        test[i] = 0;
        Vector res = sides[i] * test;
        if (res.abs() > VectorEps) {
            return true;
        }
    }
    return false;
}

Vector Bounds::minImage(Vector v) {
    for (int i=0; i<3; i++) {
        int img = round(v[i] / trace[i]);
        v -= sides[i] * (state->periodic[i] * img);
    }
    return v;
}


void export_Bounds() {
    boost::python::class_<Bounds, SHARED(Bounds)>(
        "Bounds",
        boost::python::init<SHARED(State), Vector, Vector>(
            boost::python::args("state", "lo", "hi")
        )
    )
    .def("copy", &Bounds::copy)
    .def("set", &Bounds::setPython)
    .def("getSide", &Bounds::getSide)
    .def("setSide", &Bounds::setSide)
    .def("minImage", &Bounds::minImage)
    .def_readwrite("lo", &Bounds::lo)
    .def_readwrite("hi", &Bounds::hi)
    .def_readwrite("trace", &Bounds::trace)
    ;
}
