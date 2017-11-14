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
namespace py = boost::python;
const std::string LJCHARMMType = "LJCHARMM";



FixLJCHARMM::FixLJCHARMM(boost::shared_ptr<State> state_, std::string handle_, std::string mixingRules_)
    : FixPair(state_, handle_, "all", LJCHARMMType, true, false, 1, mixingRules_),
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
    setEvalWrapper();
}

    //neighbor coefs are not used in CHARMM force field, because it specifies 1-4 sigmas and epsilons.
    //These parameters will be ignored in the evaluator
    // but we need to tell the evaluator if it's a 1-4 neighbor.  We do this by making a dummy neighborCoefs array, where all the values are 1 except the 1-4 value, which is zero.
void FixLJCHARMM::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    auto neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
                      neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), chargeRCut, virialMode, nThreadPerBlock(), nThreadPerAtom());

}

void FixLJCHARMM::singlePointEng(real *perParticleEng) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    auto neighborCoefs = state->specialNeighborCoefs;
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    //real neighborCoefs[4] = {1, 1, 1, 0}; //see comment above
    //evalWrap->energy(nAtoms,nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut);
    evalWrap->energy(nAtoms,nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, nThreadPerBlock(), nThreadPerAtom());
}


void FixLJCHARMM::singlePointEngGroupGroup(real *perParticleEng, uint32_t tagA, uint32_t tagB) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    auto neighborCoefs = state->specialNeighborCoefs;
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    //real neighborCoefs[4] = {1, 1, 1, 0}; //see comment above
    //evalWrap->energy(nAtoms,nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut);
    evalWrap->energyGroupGroup(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, tagA, tagB, nThreadPerBlock(), nThreadPerAtom());
}

void FixLJCHARMM::setEvalWrapper() {
    if (evalWrapperMode == "offload") {
        EvaluatorCHARMM eval(state->specialNeighborCoefs[2]);
        evalWrap = pickEvaluator<EvaluatorCHARMM, 5, true>(eval, chargeCalcFix);
    } else if (evalWrapperMode == "self") {
        EvaluatorCHARMM eval(state->specialNeighborCoefs[2]);
        evalWrap = pickEvaluator<EvaluatorCHARMM, 5, true>(eval, nullptr);
    }
}

bool FixLJCHARMM::prepareForRun() {
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

    auto copyEpsDiag = [&] () {
    };
    //copy in non 1-4 parameters for sig, eps

    std::vector<real> &epsPreProc = *paramMap["eps"];
    std::vector<real> &eps14PreProc = *paramMap["eps14"];

    std::vector<real> &sigPreProc = *paramMap["sig"];
    std::vector<real> &sig14PreProc = *paramMap["sig14"];
    assert(epsPreProc.size() == sigPreProc.size());
    int numTypes = state->atomParams.numTypes;
    for (int i=0; i<state->atomParams.numTypes; i++) {

        if (squareVectorRef<real>(eps14PreProc.data(), numTypes, i, i) == DEFAULT_FILL) {
            squareVectorRef<real>(eps14PreProc.data(), numTypes, i, i) = squareVectorRef<real>(epsPreProc.data(), numTypes, i, i) * state->specialNeighborCoefs[2]; 
        }

        if (squareVectorRef<real>(sig14PreProc.data(), numTypes, i, i) == DEFAULT_FILL) {
            squareVectorRef<real>(sig14PreProc.data(), numTypes, i, i) = squareVectorRef<real>(sigPreProc.data(), numTypes, i, i) * state->specialNeighborCoefs[2]; 
        }
    }


    prepareParameters(epsHandle, fillGeo, processEps, false);
    prepareParameters(eps14Handle, fillGeo, processEps, false);

	if (mixingRules==ARITHMETICTYPE) {
		prepareParameters(sigHandle, fillArith, processSig, false);
		prepareParameters(sig14Handle, fillArith, processSig, false);
	} else {
		prepareParameters(sigHandle, fillGeo, processSig, false);
		prepareParameters(sig14Handle, fillGeo, processSig, false);
	}
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);

    sendAllToDevice();
    setEvalWrapper();
    for (int i=0; i<2; i++) {
        if (state->specialNeighborCoefs[i] == state->specialNeighborCoefs[2]) {
            printf("Warning: FixLJCHARMM complains that 1-%d special neighbor coef is the same as the 1-4 coefficient.  Your 1-%d interactions will use 1-4 coefficients.\n", i+1, i+1);
        }
    }
    if (state->specialNeighborCoefs[2] == 0) {
        printf("Warning: FixLJCharmm complains that 1-4 neighbor coefficients cannot be 0\n");
        assert(state->specialNeighborCoefs[2] != 0);
    }
    prepared = true;
    return prepared;
}

std::string FixLJCHARMM::restartChunk(std::string format) {
    stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}


bool FixLJCHARMM::postRun() {

    return true;
}

void FixLJCHARMM::addSpecies(std::string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(eps14Handle, epsilons14);
    initializeParameters(sig14Handle, sigmas14);
    initializeParameters(rCutHandle, rCuts);

}

std::vector<real> FixLJCHARMM::getRCuts() { 
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

void export_FixLJCHARMM() {
    py::class_<FixLJCHARMM, boost::shared_ptr<FixLJCHARMM>, py::bases<FixPair>, boost::noncopyable > (
        "FixLJCHARMM",
        py::init<boost::shared_ptr<State>, std::string, py::optional<std::string> > (py::args("state", "handle", "mixingRules"))
    )
      ;

}
