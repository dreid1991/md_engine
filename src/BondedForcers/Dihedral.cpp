#include "Dihedral.h"
#include "State.h"
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

void Dihedral::takeIds(Dihedral *other) {
    for (int i=0; i<4; i++) {
        ids[i] = other->ids[i];
    }
}


void DihedralGPU::takeIds(Dihedral *other) {
    for (int i=0; i<4; i++) {
        ids[i] = other->ids[i];
    }
}

DihedralOPLSType::DihedralOPLSType(DihedralOPLS *dihedral) {
    for (int i=0; i<4; i++) {
        coefs[i] = dihedral->coefs[i];
    }
}

bool DihedralOPLSType::operator==(const DihedralOPLSType &other) const {
    for (int i=0; i<4; i++) {
        if (coefs[i] != other.coefs[i]) {
            return false;
        }
    }
    return true;
}

std::string Dihedral::getInfoString() {
  std::stringstream ss;
  ss << "<dihedral type=" << Dihedral::type << " />\n";
  ss << "<atomIDs>\n";
  for (int id : Dihedral::ids) {
    ss << id << "\n";
  }
  ss << "</atomIDs>\n";
  ss << "</dihedral>\n";
  return ss.str();
}

std::string DihedralOPLSType::getInfoString() {
  std::stringstream ss;
  ss << " coef_a='" << coefs[0]<< "' coef_b='" << coefs[1] << "' coef_c='" << coefs[2] << "' coef_d='" << coefs[3];
  return ss.str();
}
/*
bool DihedralOPLSType::readFromRestart(pugi::xml_node restData) {
    auto curr_param = restData.first_child();
    std::string coef_a = curr_param.attribute("coef_a").value();
    std::string coef_b = curr_param.attribute("coef_b").value();
    std::string coef_c = curr_param.attribute("coef_c").value();
    std::string coef_d = curr_param.attribute("coef_d").value();
    coefs[0] = atof(coef_a.c_str());
    coefs[1] = atof(coef_b.c_str());
    coefs[2] = atof(coef_c.c_str());
    coefs[3] = atof(coef_d.c_str());
    return true;
    }*/

std::string DihedralOPLS::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID_c='" << ids[2] << "' atomID_d='" << ids[3] << "' coef_a='" << coefs[0]<< "' coef_b='" << coefs[1] << "' coef_c='" << coefs[2] << "' coef_d='" << coefs[3] << "'/>\n";
  return ss.str();
}
/*
bool DihedralOPLS::readFromRestart(pugi::xml_node restData) {
  auto curr_param = restData.first_child();
  std::string type_ = curr_param.attribute("type").value();
  std::string atom_a = curr_param.attribute("atom_a").value();
  std::string atom_b = curr_param.attribute("atom_b").value();
  std::string atom_c = curr_param.attribute("atom_c").value();
  std::string atom_d = curr_param.attribute("atom_d").value();
  std::string coef_a = curr_param.attribute("coef_a").value();
  std::string coef_b = curr_param.attribute("coef_b").value();
  std::string coef_c = curr_param.attribute("coef_c").value();
  std::string coef_d = curr_param.attribute("coef_d").value();
  type = atoi(type_.c_str());
  ids[0] = atoi(atom_a.c_str());
  ids[1] = atoi(atom_b.c_str());
  ids[2] = atoi(atom_c.c_str());
  ids[3] = atoi(atom_d.c_str());
  coefs[0] = atof(coef_a.c_str());
  coefs[1] = atof(coef_b.c_str());
  coefs[2] = atof(coef_c.c_str());
  coefs[3] = atof(coef_d.c_str());  
  return true;
  }*/

void export_Dihedrals() {
    py::class_<DihedralOPLS, SHARED(DihedralOPLS)> ( "SimDihedralOPLS", py::init<>())
        .def_readwrite("type", &DihedralOPLS::type)
        .def_readonly("coefs", &DihedralOPLS::coefs)
        .def_readonly("ids", &DihedralOPLS::ids)
    ;

}
