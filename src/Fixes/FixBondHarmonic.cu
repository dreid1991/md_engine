#include "helpers.h"
#include "FixBondHarmonic.h"
#include "cutils_func.h"
#include "FixHelpers.h"
#include "BondEvaluate.h"
namespace py = boost::python;
using namespace std;

const std::string bondHarmonicType = "BondHarmonic";

FixBondHarmonic::FixBondHarmonic(SHARED(State) state_, string handle)
    : FixBond(state_, handle, string("None"), bondHarmonicType, true, 1),
      pyListInterface(&bonds, &pyBonds) {}



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
    ss << "<" << restartHandle << ">\n";
    for (BondVariant &bv : bonds) {
        BondHarmonic &b = get<BondHarmonic>(bv);
        //ss << b.atoms[0]->id << " " << b.atoms[1]->id << " " << b.k << " " << b.rEq << "\n";
    }
    ss << "</" << restartHandle << ">\n";
    //NOT DONE
    cout << "BOND REST CHUNK NOT DONE" << endl;
    return ss.str();
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
