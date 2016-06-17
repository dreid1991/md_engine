#include "Bond.h"
#include "boost_for_export.h"
using namespace boost::python;

bool Bond::hasAtomId(int id) {
	return ids[0] == id or ids[1] == id;
}

int Bond::otherId(int id) const {
	if (ids[0] == id) {
		return ids[1];
	} else if (ids[1] == id) {
		return ids[0];
	} 
	return -1;
}

/*
void Bond::swap() {
    Atom *x = atoms[0];
    atoms[0] = atoms[1];
    atoms[1] = x;
}
*/

std::string BondHarmonic::getInfoString() {
  std::stringstream ss;
    ss << "<members type='" << type << "' k='" << k << " rEq='" << rEq << "' atomID_a='" << ids[0] <<  "' atomID_b='" << ids[1] << "'>\n";
    return ss.str();
}

bool BondHarmonic::readFromRestart(pugi::xml_node restData) {
    auto curr_param = restData.first_child();
    std::string atom_a = curr_param.attribute("atom_a").value();
    std::string atom_b = curr_param.attribute("atom_b").value();
    std::string k_ = curr_param.attribute("k").value();
    std::string rEq_ = curr_param.attribute("thetaEq").value();
    ids[0] = atoi(atom_a.c_str());
    ids[1] = atoi(atom_b.c_str());
    k = atof(k_.c_str());
    rEq = atof(rEq_.c_str());
    return true;
}

void export_BondHarmonic() {
  
    boost::python::class_<BondHarmonic,SHARED(BondHarmonic)> ( "BondHarmonic", boost::python::init<>())
//         .def(boost::python::init<int, int ,double, double,int>())
        .def_readonly("ids", &BondHarmonic::ids)
        .def_readwrite("k", &BondHarmonic::k)
        .def_readwrite("rEq", &BondHarmonic::rEq)
    ;
}
