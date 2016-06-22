#include "Bond.h"
#include "boost_for_export.h"
using namespace boost::python;


//BondHarmonicType::BondHarmonicType(BondHarmonic *bond) {
//    k = bond->k;
//    r0 = bond->r0;
//}


bool BondHarmonicType::operator==(const BondHarmonicType &other) const {
    return k == other.k and r0 == other.r0;
}





BondHarmonic::BondHarmonic(Atom *a, Atom *b, double k_, double r0_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    k = k_;
    r0 = r0_;
    type = type_;
}
BondHarmonic::BondHarmonic(double k_, double r0_, int type_) {
    k = k_;
    r0 = r0_;
    type = type_;
}

void BondGPU::takeIds(Bond *b) { 
    myId = b->ids[0];
    otherId = b->ids[1];
}

std::string BondHarmonicType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' r0='" << r0;
  return ss.str();
}

std::string BondHarmonic::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' r0='" << r0 << "' atomID_a='" << ids[0] <<  "' atomID_b='" << ids[1] << "'/>\n";
  return ss.str();
}

void export_BondHarmonic() {
  
    boost::python::class_<BondHarmonic,SHARED(BondHarmonic)> ( "BondHarmonic", boost::python::init<>())
//         .def(boost::python::init<int, int ,double, double,int>())
        .def_readonly("ids", &BondHarmonic::ids)
        .def_readwrite("k", &BondHarmonic::k)
        .def_readwrite("r0", &BondHarmonic::r0)
    ;
}









//bond FENE
bool BondFENEType::operator==(const BondFENEType &other) const {
    return k == other.k and r0 == other.r0 and eps == other.eps and sig == other.sig;
}





BondFENE::BondFENE(Atom *a, Atom *b, double k_, double r0_, double eps_, double sig_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    k = k_;
    r0 = r0_;
    eps = eps_;
    sig = sig_;
    type = type_;
}
BondFENE::BondFENE(double k_, double r0_, double eps_, double sig_, int type_) {
    k = k_;
    r0 = r0_;
    eps = eps_;
    sig = sig_;
    type = type_;
}

std::string BondFENEType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' r0='" << r0 << "' eps='" << eps << "' sig='" << sig;
  return ss.str();
}

std::string BondFENE::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' r0='" << r0 << "' eps='" << eps << "' sig='" << sig << "' atomID_a='" << ids[0] <<  "' atomID_b='" << ids[1] << "'/>\n";
  return ss.str();
}

void export_BondFENE() {
  
    boost::python::class_<BondFENE,SHARED(BondFENE)> ( "BondFENE", boost::python::init<>())
//         .def(boost::python::init<int, int ,double, double,int>())
        .def_readonly("ids", &BondFENE::ids)
        .def_readwrite("k", &BondFENE::k)
        .def_readwrite("r0", &BondFENE::r0)
        .def_readwrite("eps", &BondFENE::eps)
        .def_readwrite("sig", &BondFENE::sig)
    ;
}
