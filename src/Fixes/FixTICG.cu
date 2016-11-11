#include "FixTICG.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"
#include "EvaluatorWrapper.h"

const std::string TICGType = "TICG";

FixTICG::FixTICG(boost::shared_ptr<State> state_, std::string handle_)
  : FixPair(state_, handle_, "all", TICGType, true, false, 1),
    CHandle("C"),  rCutHandle("rCut")
{
    initializeParameters(CHandle, Cs);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {rCutHandle, CHandle};
    readFromRestart();
}

void FixTICG::compute(bool computeVirials) {

    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
                      neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), chargeRCut, computeVirials);

}

void FixTICG::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energy(nAtoms, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut);





}

bool FixTICG::prepareForRun() {
    //loop through all params and fill with appropriate lambda function, then send all to device
    auto none = [] (float a){};

	std::function<float(float)> processCs = [] (float a) {
        return a;
    };
	std::function<float(float)> processRCut = [] (float a) {
        return a*a;
    };
    prepareParameters(CHandle,  processCs);
    prepareParameters(rCutHandle, processRCut);
    sendAllToDevice();
    setEvalWrapper();
    return true;
}

void FixTICG::setEvalWrapper() {
    EvaluatorTICG eval;
    evalWrap = pickEvaluator<EvaluatorTICG, 2>(eval, chargeCalcFix);
}
std::string FixTICG::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}

bool FixTICG::postRun() {
    return true;
}

void FixTICG::addSpecies(std::string handle) {
    initializeParameters(CHandle, Cs);
    initializeParameters(rCutHandle, rCuts);

}

std::vector<float> FixTICG::getRCuts() {  
    std::vector<float> res;
    std::vector<float> &src = *(paramMap[rCutHandle]);
    for (float x : src) {
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
        boost::python::init<boost::shared_ptr<State>, std::string> (
            boost::python::args("state", "handle"))
    );

}
