#include "FixLJCHARMM.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "State.h"
#include "cutils_func.h"
#include "ReadConfig.h"
#include "EvaluatorWrapper.h"
#include "PairEvaluatorCHARMM.h"
#include "EvaluatorWrapper.h"
//#include "ChargeEvaluatorEwald.h"
using namespace std;
namespace py = boost::python;
const string LJCHARMMType = "LJCHARMM";



FixLJCHARMM::FixLJCHARMM(boost::shared_ptr<State> state_, string handle_)
    : FixPair(state_, handle_, "all", LJCHARMMType, true, false, 1),
    epsHandle("eps"), sigHandle("sig"), eps14Handle("eps14"), sig14Handle("sig14"), rCutHandle("rCut")
{

    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(eps14Handle, epsilons14);
    initializeParameters(sig14Handle, sigmas14);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {rCutHandle, epsHandle, sigHandle, eps14Handle, sig14Handle};
    readFromRestart();
    canAcceptChargePairCalc = true;
    setEvalW
    origEvalWrap = getEvalWrapper();
}

    //neighbor coefs are not used in CHARMM force field, because it specifies 1-4 sigmas and epsilons.
    //These parameters will be ignored in the evaluator
    // but we need to tell the evaluator if it's a 1-4 neighbor.  We do this by making a dummy neighborCoefs array, where all the values are 1 except the 1-4 value, which is zero.
void FixLJCHARMM::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float neighborCoefs[4] = {1, 1, 1, 0};//see comment above

    evalWrap->compute(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
                      neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), chargeRCut, computeVirials);

}

void FixLJCHARMM::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float neighborCoefs[4] = {1, 1, 1, 0}; //see comment above
    evalWrap->energy(nAtoms, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut);



}

void FixLJCHARMM::setEvalWrapper() {
    EvaluatorCHARMM eval;
    evalWrap = pickEvaluator<EvaluatorCHARMM, 3, true>(eval, chargeCalcFix);

}

bool FixLJCHARMM::prepareForRun() {
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

    auto copyEpsDiag = [&] () {
    };
    //copy in non 1-4 parameters for sig, eps

    std::vector<float> &epsPreProc = *paramMap["eps"];
    std::vector<float> &sigPreProc = *paramMap["sig"];

    std::vector<float> &eps14PreProc = *paramMap["eps14"];
    std::vector<float> &sig14PreProc = *paramMap["sig14"];
    assert(epsPreProc.size() == sigPreProc.size());
    for (int i=0; i<epsPreProc.size(); i++) {
        if (eps14PreProc[i] == DEFAULT_FILL) {
            eps14PreProc[i] = epsPreProc[i]; //times some coefficient?
        }
        if (sig14PreProc[i] == DEFAULT_FILL) {
            sig14PreProc[i] = sigPreProc[i]; 
        }
    }


    prepareParameters(epsHandle, fillEps, processEps, false);
    prepareParameters(sigHandle, fillSig, processSig, false);
    prepareParameters(eps14Handle, fillEps, processEps, false);
    prepareParameters(sig14Handle, fillSig, processSig, false);
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);

    sendAllToDevice();
    setEvalWrapper();
    return true;
}

string FixLJCHARMM::restartChunk(string format) {
    stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}


bool FixLJCHARMM::postRun() {

    return true;
}

void FixLJCHARMM::addSpecies(string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(eps14Handle, epsilons14);
    initializeParameters(sig14Handle, sigmas14);
    initializeParameters(rCutHandle, rCuts);

}

vector<float> FixLJCHARMM::getRCuts() { 
    vector<float> res;
    vector<float> &src = *(paramMap[rCutHandle]);
    for (float x : src) {
        if (x == DEFAULT_FILL) {
            res.push_back(-1);
        } else {
            res.push_back(x);
        }
    }

    return res;
}

void export_FixLJCHARMM() {
    py::class_<FixLJCHARMM, boost::shared_ptr<FixLJCHARMM>, py::bases<FixPair>, boost::noncopyable > (
        "FixLJCHARMM",
        py::init<boost::shared_ptr<State>, string> (py::args("state", "handle"))
    )
      ;

}
