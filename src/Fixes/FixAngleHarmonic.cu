
#include "FixHelpers.h"
#include "helpers.h"
#include "FixAngleHarmonic.h"
#include "cutils_func.h"
#include "AngleEvaluate.h"
using namespace std;
const string angleHarmonicType = "AngleHarmonic";
FixAngleHarmonic::FixAngleHarmonic(boost::shared_ptr<State> state_, string handle)
  : FixPotentialMultiAtom(state_, handle, angleHarmonicType, true),
    pyListInterface(&forcers, &pyForcers)
{   }

namespace py = boost::python;

void FixAngleHarmonic::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    compute_force_angle<<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleGPU) * maxForcersPerBlock + parameters.size() * sizeof(AngleHarmonicType)>>>(nAtoms, state->gpd.xs(activeIdx), state->gpd.fs(activeIdx), state->gpd.idToIdxs.getTex(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), evaluator);

}

void FixAngleHarmonic::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    compute_energy_angle<<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleGPU) * maxForcersPerBlock + parameters.size() * sizeof(AngleHarmonicType)>>>(nAtoms, state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.getTex(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), evaluator);
}
//void cumulativeSum(int *data, int n);
// okay, so the net result of this function is that two arrays (items, idxs of
// items) are on the gpu and we know how many bonds are in bondiest block

void FixAngleHarmonic::createAngle(Atom *a, Atom *b, Atom *c, double k, double thetaEq, int type) {
    vector<Atom *> atoms = {a, b, c};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=COEF_DEFAULT and thetaEq!=COEF_DEFAULT);
    }
    forcers.push_back(AngleHarmonic(a, b, c, k, thetaEq, type));
    pyListInterface.updateAppendedMember();
}

void FixAngleHarmonic::setAngleTypeCoefs(int type, double k, double thetaEq) {
    //cout << type << " " << k << " " << thetaEq << endl;
    assert(thetaEq>=0);
    AngleHarmonic dummy(k, thetaEq);
    setForcerType(type, dummy);
}

string FixAngleHarmonic::restartChunk(string format) {
    stringstream ss;
    return ss.str();
}

void export_FixAngleHarmonic() {
    boost::python::class_<FixAngleHarmonic,
                          boost::shared_ptr<FixAngleHarmonic>,
                          boost::python::bases<Fix, TypedItemHolder> >(
        "FixAngleHarmonic",
        boost::python::init<boost::shared_ptr<State>, string>(
                                boost::python::args("state", "handle"))
    )
    .def("createAngle", &FixAngleHarmonic::createAngle,
            (boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("thetaEq")=COEF_DEFAULT,
             boost::python::arg("type")=-1)
        )
    .def("setAngleTypeCoefs", &FixAngleHarmonic::setAngleTypeCoefs,
            (boost::python::arg("type")=-1,
             boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("thetaEq")=COEF_DEFAULT
            )
        )
    .def_readonly("angles", &FixAngleHarmonic::pyForcers)
    ;
}

