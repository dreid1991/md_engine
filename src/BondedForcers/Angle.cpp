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

void Angle::takeIds(Angle *other) {
    for (int i=0; i<3; i++) {
        ids[i] = other->ids[i];
    }
}

AngleHarmonic::AngleHarmonic(double k_, double thetaEq_, int type_) {
    for (int i=0; i<3; i++) {
        ids[i] = -1;
    }
    k = k_;
    thetaEq = thetaEq_;
    type = type_;
}


void AngleGPU::takeIds(Angle *other) {
    for (int i=0; i<3; i++) {
        ids[i] = other->ids[i];
    }
}


AngleHarmonicType::AngleHarmonicType(AngleHarmonic *angle) {
    k = angle->k;
    thetaEq = angle->thetaEq;
}

std::string AngleHarmonicType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' thetaEq='" << thetaEq;
  return ss.str();
}

bool AngleHarmonicType::readFromRestart(pugi::xml_node restData) {
    auto curr_param = restData.first_child();
    std::string k_ = curr_param.attribute("k").value();
    std::string thetaEq_ = curr_param.attribute("thetaEq").value();
    k = atof(k_.c_str());
    thetaEq = atof(thetaEq_.c_str());
    return true;
}

std::string AngleHarmonic::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' thetaEq='" << thetaEq << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID_c='" << ids[2] << "'/>\n";
  return ss.str();
}

bool AngleHarmonic::readFromRestart(pugi::xml_node restData) {
    auto curr_param = restData.first_child();
    std::string type_ = curr_param.attribute("type").value();
    std::string atom_a = curr_param.attribute("atom_a").value();
    std::string atom_b = curr_param.attribute("atom_b").value();
    std::string atom_c = curr_param.attribute("atom_c").value();
    std::string k_ = curr_param.attribute("k").value();
    std::string thetaEq_ = curr_param.attribute("thetaEq").value();
    type = atoi(type_.c_str());
    ids[0] = atoi(atom_a.c_str());
    ids[1] = atoi(atom_b.c_str());
    ids[2] = atoi(atom_c.c_str());
    k = atof(k_.c_str());
    thetaEq = atof(thetaEq_.c_str());
    return true;
}

bool AngleHarmonicType::operator==(const AngleHarmonicType &other) const {
    return k == other.k and thetaEq == other.thetaEq;
}
void export_AngleHarmonic() {
    boost::python::class_<AngleHarmonic, SHARED(AngleHarmonic)> ( "AngleHarmonic", boost::python::init<>())
        .def_readwrite("thetaEq", &AngleHarmonic::thetaEq)
        .def_readwrite("k", &AngleHarmonic::k)
        .def_readwrite("type", &AngleHarmonic::type)
        .def_readonly("ids", &AngleHarmonic::ids)

    ;
}
