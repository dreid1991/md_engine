#include "Bond.h"
#include "boost_for_export.h"
using namespace boost::python;


BondHarmonicType::BondHarmonicType(BondHarmonic *bond) {
    k = bond->k;
    rEq = bond->rEq;
}


bool BondHarmonicType::operator==(const BondHarmonicType &other) const {
    return k == other.k and rEq == other.rEq;
}





BondHarmonic::BondHarmonic(Atom *a, Atom *b, double k_, double rEq_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    k = k_;
    rEq = rEq_;
    type = type_;
}
BondHarmonic::BondHarmonic(double k_, double rEq_, int type_) {
    k = k_;
    rEq = rEq_;
    type = type_;
}

void BondGPU::takeIds(Bond *b) { 
    myId = b->ids[0];
    otherId = b->ids[0];
}

std::string BondHarmonicType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' rEq='" << rEq;
  return ss.str();
}

std::string BondHarmonic::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << " rEq='" << rEq << "' atomID_a='" << ids[0] <<  "' atomID_b='" << ids[1] << "'>\n";
  return ss.str();
}

void export_BondHarmonic() {
  
    boost::python::class_<BondHarmonic,SHARED(BondHarmonic)> ( "BondHarmonic", boost::python::init<>())
//         .def(boost::python::init<int, int ,double, double,int>())
        .def_readonly("ids", &BondHarmonic::ids)
        .def_readwrite("k", &BondHarmonic::k)
        .def_readwrite("rEq", &BondHarmonic::rEq)
    ;
}
