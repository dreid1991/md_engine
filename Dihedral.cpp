#include "Dihedral.h"
#include "boost_for_export.h"
#include "array_indexing_suite.hpp"
namespace py = boost::python;
DihedralOPLS::DihedralOPLS(Atom *a, Atom *b, Atom *c, Atom *d, double coefs_[4], int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    ids[2] = c->id;
    ids[3] = d->id;
    for (int i=0; i<4; i++) {
        coefs[i] = coefs_[i];
    }
    type = type_;
}

DihedralOPLS::DihedralOPLS(double coefs_[4], int type_) {
    for (int i=0; i<4; i++) {
        ids[i] = -1;
    }
    for (int i=0; i<4; i++) {
        coefs[i] = coefs_[i];
    }
    type = type_;
}

void DihedralOPLS::takeParameters(DihedralOPLS &other) {
    for (int i=0; i<4; i++) {
        coefs[i] = other.coefs[i];
    }
}

void DihedralOPLS::takeIds(DihedralOPLS &other) {
    for (int i=0; i<4; i++) {
        ids[i] = other.ids[i];
    }
}

void DihedralOPLSGPU::takeParameters(DihedralOPLS &other) {
    for (int i=0; i<4; i++) {
        coefs[i] = other.coefs[i];
    }
}

void DihedralOPLSGPU::takeIds(DihedralOPLS &other) {
    for (int i=0; i<4; i++) {
        ids[i] = other.ids[i];
    }
}




void export_Dihedrals() {
    py::class_<DihedralOPLS, SHARED(DihedralOPLS)> ( "SimDihedralOPLS", py::init<>())
        .def_readwrite("type", &DihedralOPLS::type)
        .def_readonly("coefs", &DihedralOPLS::coefs)
        .def_readonly("ids", &DihedralOPLS::ids)

    ;

}
