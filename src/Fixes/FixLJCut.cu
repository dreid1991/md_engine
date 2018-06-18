#include "FixLJCut.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "State.h"
#include "cutils_func.h"
#include "ReadConfig.h"
#include "EvaluatorWrapper.h"
//#include "ChargeEvaluatorEwald.h"
namespace py = boost::python;
const std::string LJCutType = "LJCut";



FixLJCut::FixLJCut(boost::shared_ptr<State> state_, std::string handle_, std::string mixingRules_)
    : FixPair(state_, handle_, "all", LJCutType, true, false, 1, mixingRules_),
    epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut")
{

    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {rCutHandle, epsHandle, sigHandle};
    readFromRestart();
    canAcceptChargePairCalc = true;
    setEvalWrapper();
}

void FixLJCut::compute(int virialMode) {
    int nAtoms       = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    real *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms, nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.neighborlistPositions.data(),grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
                      neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), chargeRCut, virialMode, nThreadPerBlock(), nThreadPerAtom());

}

void FixLJCut::singlePointEng(real *perParticleEng) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    real *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energy(nAtoms, nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, nThreadPerBlock(), nThreadPerAtom());
}

void FixLJCut::singlePointEngGroupGroup(real *perParticleEng, uint32_t tagA, uint32_t tagB) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    real *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energyGroupGroup(nAtoms, nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, tagA, tagB, nThreadPerBlock(), nThreadPerAtom());
}

void FixLJCut::setEvalWrapper() {
    if (evalWrapperMode == "offload") {
        EvaluatorLJ eval;
        evalWrap = pickEvaluator<EvaluatorLJ, 3, true>(eval, chargeCalcFix);
    } else if (evalWrapperMode == "self") {
        EvaluatorLJ eval;
        evalWrap = pickEvaluator<EvaluatorLJ, 3, true>(eval, nullptr);
    } else {
        mdError("evalWrapperMode in FixLJCut is neither offload nor self; aborting.");
    }
    
}

bool FixLJCut::prepareForRun() {
    //loop through all params and fill with appropriate lambda function, then send all to device
    auto fillGeo = [] (real a, real b) {
        return sqrt(a*b);
    };

    auto fillArith = [] (real a, real b) {
        return (a+b) / 2.0;
    };
    auto fillRCut = [this] (real a, real b) {
        return (real) std::fmax(a, b);
    };
    auto none = [] (real a){};

    auto fillRCutDiag = [this] () {
        return (real) state->rCut;
    };

    auto processEps = [] (real a) {
        return 24*a;
    };
    auto processSig = [] (real a) {
        return pow(a, 6);
    };
    auto processRCut = [] (real a) {
        return a*a;
    };
    prepareParameters(epsHandle, fillGeo, processEps, false);
	if (mixingRules==ARITHMETICTYPE) {
		prepareParameters(sigHandle, fillArith, processSig, false);
	} else {
		prepareParameters(sigHandle, fillGeo, processSig, false);
	}
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);

    sendAllToDevice();
    setEvalWrapper();
    prepared = true;
    return prepared;
}

std::string FixLJCut::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}



void FixLJCut::addSpecies(std::string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);

}

std::vector<real> FixLJCut::getRCuts() { 
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

void export_FixLJCut() {
    py::class_<FixLJCut, boost::shared_ptr<FixLJCut>, py::bases<FixPair>, boost::noncopyable > (
        "FixLJCut",
        py::init<boost::shared_ptr<State>, std::string, py::optional<std::string> > (py::args("state", "handle", "mixingRules"))
    )
      ;

}
