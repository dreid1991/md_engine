#include "FixLJCut.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"

const std::string LJCutType = "LJCut";

FixLJCut::FixLJCut(SHARED(State) state_, string handle_) : FixPair(state_, handle_, "all", LJCutType, 1), epsHandle("eps"), sigHandle("sig"), rCutHandle("r\
Cut") {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {epsHandle, sigHandle, rCutHandle};
    forceSingle = true;

}
void FixLJCut::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

        compute_force_iso<EvaluatorLJ, 3>  <<<NBLOCK(nAtoms), PERBLOCK, 3*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), neig\
hborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU\
, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], evaluator);



}

void FixLJCut::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    compute_energy_iso<EvaluatorLJ, 3><<<NBLOCK(nAtoms), PERBLOCK, 3*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), perParticleEng, neighbor\
Counts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, ne\
ighborCoefs[0], neighborCoefs[1], neighborCoefs[2], evaluator);



}

bool FixLJCut::prepareForRun() {
    //loop through all params and fill with appropriate lambda function, then send all to device
    auto fillEps = [] (float a, float b) {
        return sqrt(a*b);
    };

    auto fillSig = [] (float a, float b) {
        return (a+b) / 2.0;
    };
    auto fillRCut = [this] (float a, float b) {
        return (float) std::fmax(a, b);
    };
    auto none = [] (float a){};

    auto fillRCutDiag = [this] () {
        return (float) state->rCut;
    };

    auto processEps = [] (float a) {
        return 24*a;
    };
    auto processSig = [] (float a) {
        return pow(a, 6);
    };
    auto processRCut = [] (float a) {
        return a*a;
    };
    prepareParameters(epsHandle, fillEps, processEps, false);
    prepareParameters(sigHandle, fillSig, processSig, false);
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);
    sendAllToDevice();
    return true;
}

string FixLJCut::restartChunk(string format) {
    //test this
    stringstream ss;
    ss << "<" << restartHandle << ">\n";
    ss << restartChunkPairParams(format);
    ss << "</" << restartHandle << ">\n";
    return ss.str();
}

bool FixLJCut::readFromRestart(pugi::xml_node restData) {
    vector<float> epsilons = xml_readNums<float>(restData, epsHandle);
    initializeParameters(epsHandle, epsilons);
    vector<float> sigmas = xml_readNums<float>(restData, sigHandle);
    initializeParameters(sigHandle, sigmas);
    vector<float> rCuts = xml_readNums<float>(restData, rCutHandle);
    initializeParameters(rCutHandle, rCuts);
    cout << "Reading LJ parameters from restart\n";
    return true;

}
void FixLJCut::postRun() {
    resetToPreproc(sigHandle);
    resetToPreproc(epsHandle);
    resetToPreproc(rCutHandle);
}

void FixLJCut::addSpecies(string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);

}

vector<float> FixLJCut::getRCuts() { //to be called after prepare.  These are squares now
    return LISTMAP(float, float, rc, rCuts, sqrt(rc));
}

void export_FixLJCut() {
    boost::python::class_<FixLJCut,
                          SHARED(FixLJCut),
                          boost::python::bases<FixPair>, boost::noncopyable > (
        "FixLJCut",
        boost::python::init<SHARED(State), string> (
            boost::python::args("state", "handle"))
    );

}