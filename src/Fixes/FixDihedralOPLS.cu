#include "helpers.h"
#include "FixDihedralOPLS.h"
#include "FixHelpers.h"
#include "cutils_func.h"
#include "DihedralEvaluate.h"
namespace py = boost::python;
using namespace std;

const std::string dihedralOPLSType = "DihedralOPLS";


FixDihedralOPLS::FixDihedralOPLS(SHARED(State) state_, string handle) : FixPotentialMultiAtom (state_, handle, dihedralOPLSType, true){}


void FixDihedralOPLS::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();


    compute_force_dihedral<<<NBLOCK(nAtoms), PERBLOCK, sizeof(DihedralGPU) * maxForcersPerBlock + sizeof(DihedralOPLSType) * parameters.size() >>>(nAtoms, state->gpd.xs(activeIdx), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), evaluator);

}

void FixDihedralOPLS::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();


    compute_energy_dihedral<<<NBLOCK(nAtoms), PERBLOCK, sizeof(DihedralGPU) * maxForcersPerBlock + sizeof(DihedralOPLSType) * parameters.size() >>>(nAtoms, state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.getTex(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), evaluator);

}



void FixDihedralOPLS::createDihedral(Atom *a, Atom *b, Atom *c, Atom *d, double v1, double v2, double v3, double v4, int type) {
    double vs[4] = {v1, v2, v3, v4};
    if (type==-1) {
        for (int i=0; i<4; i++) {
            assert(vs[i] != COEF_DEFAULT);
        }
    }
    forcers.push_back(DihedralOPLS(a, b, c, d, vs, type));
    pyListInterface.updateAppendedMember();
}


void FixDihedralOPLS::createDihedralPy(Atom *a, Atom *b, Atom *c, Atom *d, py::list coefs, int type) {
    double coefs_c[4];
    if (type!=-1) {
        createDihedral(a, b, c, d, COEF_DEFAULT, COEF_DEFAULT, COEF_DEFAULT, COEF_DEFAULT, type);
    } else {
        assert(len(coefs) == 4);
        for (int i=0; i<4; i++) {
            py::extract<double> coef(coefs[i]);
            assert(coef.check());
            coefs_c[i] = coef;
        }
        createDihedral(a, b, c, d, coefs_c[0], coefs_c[1], coefs_c[2], coefs_c[3], type);

    }
}

void FixDihedralOPLS::setDihedralTypeCoefs(int type, py::list coefs) {
    assert(len(coefs)==4);
    double coefs_c[4];
    for (int i=0; i<4; i++) {
        py::extract<double> coef(coefs[i]);
        assert(coef.check());
        coefs_c[i] = coef;
    }

    DihedralOPLS dummy(coefs_c, type);
    setForcerType(type, dummy);
}



string FixDihedralOPLS::restartChunk(string format) {
    stringstream ss;

    return ss.str();
}

void export_FixDihedralOPLS() {
    py::class_<FixDihedralOPLS,
                          SHARED(FixDihedralOPLS),
                          py::bases<Fix, TypedItemHolder> > (
        "FixDihedralOPLS",
        py::init<SHARED(State), string> (
            py::args("state", "handle")
        )
    )
    .def("createDihedral", &FixDihedralOPLS::createDihedralPy,
            (py::arg("coefs")=py::list(),
             py::arg("type")=-1)
        )

    .def("setDihedralTypeCoefs", &FixDihedralOPLS::setDihedralTypeCoefs, 
            (py::arg("type"), 
             py::arg("coefs"))
            )
    .def_readonly("dihedrals", &FixDihedralOPLS::pyForcers)

    ;

}

