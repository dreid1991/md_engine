#include "helpers.h"
#include "FixBondHarmonic.h"
#include "cutils_func.h"
#include "FixHelpers.h"
#include "BondEvaluate.h"
#include "ReadConfig.h"
namespace py = boost::python;
using namespace std;

const std::string bondHarmonicType = "BondHarmonic";

FixBondHarmonic::FixBondHarmonic(SHARED(State) state_, string handle)
    : FixBond(state_, handle, string("None"), bondHarmonicType, true, 1) {
  if (state->readConfig->fileOpen) {
    auto restData = state->readConfig->readFix(type, handle);
    if (restData) {
      std::cout << "Reading restart data for fix " << handle << std::endl;
      readFromRestart(restData);
    }
  }
}



void FixBondHarmonic::createBond(Atom *a, Atom *b, double k, double rEq, int type) {
    vector<Atom *> atoms = {a, b};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=-1 and rEq!=-1);
    }
    bonds.push_back(BondHarmonic(a, b, k, rEq, type));
    pyListInterface.updateAppendedMember();
    
}

void FixBondHarmonic::setBondTypeCoefs(int type, double k, double rEq) {
    assert(rEq>=0);
    BondHarmonic dummy(k, rEq, type);
    setBondType(type, dummy);
}

void FixBondHarmonic::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    //cout << "Max bonds per block is " << maxBondsPerBlock << endl;
    compute_force_bond<<<NBLOCK(nAtoms), PERBLOCK, sizeof(BondGPU) * maxBondsPerBlock + sizeof(BondHarmonicType) * parameters.size()>>>(nAtoms, state->gpd.xs(activeIdx), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), bondsGPU.data(), bondIdxs.data(), parameters.data(), parameters.size(), state->boundsGPU, evaluator);
}

void FixBondHarmonic::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    //cout << "Max bonds per block is " << maxBondsPerBlock << endl;
    compute_energy_bond<<<NBLOCK(nAtoms), PERBLOCK, sizeof(BondGPU) * maxBondsPerBlock + sizeof(BondHarmonicType) * parameters.size()>>>(nAtoms, state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.getTex(), bondsGPU.data(), bondIdxs.data(), parameters.data(), parameters.size(), state->boundsGPU, evaluator);
}

string FixBondHarmonic::restartChunk(string format) {
  stringstream ss;
  ss << "<types>\n";
  for (auto it = bondTypes.begin(); it != bondTypes.end(); it++) {
    ss << "<" << "type id='" << it->first << "'";
    ss << bondTypes[it->first].getInfoString() << "'/>\n";
  }
  ss << "</types>\n";
  ss << "<members>\n";
  for (BondVariant &forcerVar : bonds) {
    BondHarmonic &forcer= boost::get<BondHarmonic>(forcerVar);
    ss << forcer.getInfoString();
  }
  ss << "</members>\n";
  return ss.str();
}

bool FixBondHarmonic::readFromRestart(pugi::xml_node restData) {
  auto curr_node = restData.first_child();
  while (curr_node) {
    std::string tag = curr_node.name();
    if (tag == "types") {
      for (auto type_node = curr_node.first_child(); type_node; type_node = type_node.next_sibling()) {
        int type;
        double k;
        double rEq;
	std::string type_ = type_node.attribute("id").value();
        type = atoi(type_.c_str());
	std::string k_ = type_node.attribute("k").value();
	std::string rEq_ = type_node.attribute("rEq").value();
        k = atof(k_.c_str());
        rEq = atof(rEq_.c_str());

        setBondTypeCoefs(type, k, rEq);
      }
    } else if (tag == "members") {
      for (auto member_node = curr_node.first_child(); member_node; member_node = member_node.next_sibling()) {
        int type;
        double k;
        double rEq;
        int ids[2];
	std::string type_ = member_node.attribute("type").value();
	std::string atom_a = member_node.attribute("atom_a").value();
	std::string atom_b = member_node.attribute("atom_b").value();
	std::string k_ = member_node.attribute("k").value();
	std::string rEq_ = member_node.attribute("rEq").value();
        type = atoi(type_.c_str());
        ids[0] = atoi(atom_a.c_str());
        ids[1] = atoi(atom_b.c_str());
        Atom * a = state->idToAtom(ids[0]);
        Atom * b = state->idToAtom(ids[1]);
        k = atof(k_.c_str());
        rEq = atof(rEq_.c_str());

        createBond(a, b, k, rEq, type);
      }
    }
    curr_node = curr_node.next_sibling();
  }
  return true;
}

void export_FixBondHarmonic() {
  

  
    py::class_<FixBondHarmonic, SHARED(FixBondHarmonic), py::bases<Fix, TypedItemHolder> >
    (
        "FixBondHarmonic", py::init<SHARED(State), string> (py::args("state", "handle"))
    )
    .def("createBond", &FixBondHarmonic::createBond,
            (py::arg("k")=-1,
             py::arg("rEq")=-1,
             py::arg("type")=-1)
        )
    .def("setBondTypeCoefs", &FixBondHarmonic::setBondTypeCoefs,
            (py::arg("type"),
             py::arg("k"),
             py::arg("rEq"))
        )
    .def_readonly("bonds", &FixBondHarmonic::pyBonds)    
    ;

}
