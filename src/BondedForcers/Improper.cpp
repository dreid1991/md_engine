#include "Improper.h"
#include "boost_for_export.h"
namespace py = boost::python;

ImproperHarmonic::ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k_, double thetaEq_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    ids[2] = c->id;
    ids[3] = d->id;
    k = k_;
    thetaEq = thetaEq_;
    type = type_;

}
ImproperHarmonic::ImproperHarmonic(double k_, double thetaEq_, int type_) {
    for (int i=0; i<4; i++) {
        ids[i] = -1;
    }
    k = k_;
    thetaEq = thetaEq_;
    type = type_;

}

void Improper::takeIds(Improper *other) {
    for (int i=0; i<4; i++) {
        ids[i] = other->ids[i];
    }
}



void ImproperGPU::takeIds(Improper *other) {
    for (int i=0; i<4; i++) {
        ids[i] = other->ids[i];
    }
}

ImproperHarmonicType::ImproperHarmonicType(ImproperHarmonic *imp) {
    k = imp->k;
    thetaEq = imp->thetaEq;
}
bool ImproperHarmonicType::operator==(const ImproperHarmonicType &other) const {
    return k == other.k and thetaEq == other.thetaEq;
}

std::string ImproperHarmonic::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' thetaEq='" << thetaEq << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID\
_c='" << ids[2] << "' atomID_d='" << ids[3] << "'/>\n";
  return ss.str();
}

std::string ImproperHarmonicType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' thetaEq='" << thetaEq;
  return ss.str();
}

void export_Impropers() {
    py::class_<ImproperHarmonic, SHARED(ImproperHarmonic)> ( "SimImproperHarmonic", py::init<>())
        .def_readwrite("type", &ImproperHarmonic::type)
        .def_readonly("thetaEq", &ImproperHarmonic::thetaEq)
        .def_readonly("k", &ImproperHarmonic::k)
        .def_readonly("ids", &ImproperHarmonic::ids)

    ;

}
