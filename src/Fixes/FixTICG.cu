#include "FixTICG.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"
#include "EvaluatorWrapper.h"

const std::string TICGType = "TICG";

FixTICG::FixTICG(boost::shared_ptr<State> state_, std::string handle_, std::string mixingRules_)
  : FixPair(state_, handle_, "all", TICGType, true, false, 1, mixingRules_),
    CHandle("C"),  rCutHandle("rCut")
{
    initializeParameters(CHandle, Cs);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {rCutHandle, CHandle};
    readFromRestart();
    setEvalWrapper();
}

void FixTICG::compute(int virialMode) {

    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    real *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(),  grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
                      neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), chargeRCut, virialMode, nThreadPerBlock(), nThreadPerAtom());


}

void FixTICG::singlePointEng(real *perParticleEng) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    real *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energy(nAtoms,nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, nThreadPerBlock(), nThreadPerAtom());






}

bool FixTICG::prepareForRun() {
    //loop through all params and fill with appropriate lambda function, then send all to device
    auto none = [] (real a){};

	std::function<real(real)> processCs = [] (real a) {
        return a;
    };
	std::function<real(real)> processRCut = [] (real a) {
        return a*a;
    };
    prepareParameters(CHandle,  processCs);
    prepareParameters(rCutHandle, processRCut);
    sendAllToDevice();
    setEvalWrapper();
    prepared = true;
    return prepared;
}

void FixTICG::setEvalWrapper() {
    if (evalWrapperMode == "orig") {
        EvaluatorTICG eval;
        evalWrap = pickEvaluator<EvaluatorTICG, 2, true>(eval, chargeCalcFix);
     } else if (evalWrapperMode == "self") {
        EvaluatorTICG eval;
        evalWrap = pickEvaluator<EvaluatorTICG, 2, true>(eval, nullptr);
     }
}
std::string FixTICG::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}


void FixTICG::addSpecies(std::string handle) {
    initializeParameters(CHandle, Cs);
    initializeParameters(rCutHandle, rCuts);

}

std::vector<real> FixTICG::getRCuts() {  
    std::vector<real> res;
    std::vector<real> &src = *(paramMap[rCutHandle]);
    for (real x : src) {
        if (x == DEFAULT_FILL) {
            res.push_back(-1);
        } else {
            res.push_back(x);
        }
    }

    return res;
}

void export_FixTICG() {
    boost::python::class_<FixTICG,
                          boost::shared_ptr<FixTICG>,
                          boost::python::bases<FixPair>, boost::noncopyable > (
        "FixTICG",
        boost::python::init<boost::shared_ptr<State>, std::string, boost::python::optional<std::string> > (
            boost::python::args("state", "handle", "mixingRules"))
    );

}
