#include "FixLJCutFS.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"
#include "EvaluatorWrapper.h"

const std::string LJCutType = "LJCutFS";
namespace py = boost::python;

FixLJCutFS::FixLJCutFS(boost::shared_ptr<State> state_, std::string handle_, std::string mixingRules_)
    : FixPair(state_, handle_, "all", LJCutType, true, false, 1, mixingRules_),
      epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut") {
 
    // FixPair::initializeParameters()
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    initializeParameters("FCutHandle",FCuts);
    paramOrder = {rCutHandle, epsHandle, sigHandle,"FCutHandle"};

    canAcceptChargePairCalc = true;
    setEvalWrapper();
}

void FixLJCutFS::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    real *neighborCoefs = state->specialNeighborCoefs;
    
    evalWrap->compute(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
                      neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), chargeRCut, virialMode, nThreadPerBlock(), nThreadPerAtom());
}

void FixLJCutFS::singlePointEng(real *perParticleEng) {
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

void FixLJCutFS::singlePointEngGroupGroup(real *perParticleEng, uint32_t tagA, uint32_t tagB) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    real *neighborCoefs = state->specialNeighborCoefs;

    evalWrap->energyGroupGroup(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, tagA, tagB, nThreadPerBlock(), nThreadPerAtom());

}

bool FixLJCutFS::prepareForRun() {
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
    
    std::function<real(int,int)>fillFCut = [this] (int a, int b) {
        int numTypes = state->atomParams.numTypes;
        real epstimes24 = 24.0 * squareVectorRef<real>(paramMap[epsHandle]->data(),numTypes,a,b);
        real rCutSqr    = pow(squareVectorRef<real>(paramMap[rCutHandle]->data(),numTypes,a,b),2);
        real sig6 = pow(squareVectorRef<real>(paramMap[sigHandle]->data(),numTypes,a,b),6);
        real p1 = epstimes24 * 2.0 * sig6 * sig6;
        real p2 = epstimes24 * sig6;
        real r2Inv = 1.0 / rCutSqr;
        real rInv = sqrt(r2Inv);
        real r6Inv = r2Inv * r2Inv * r2Inv;
        real forceScalar = r6Inv * rInv * (p1 * r6Inv - p2);
        return forceScalar;
    };
    //paramOrder = {rCutHandle, epsHandle, sigHandle, "FCutHandle"};
    
    prepareParameters(epsHandle, fillGeo, processEps, false);
	if (mixingRules==ARITHMETICTYPE) {
		prepareParameters(sigHandle, fillArith, processSig, false);
	} else {
		prepareParameters(sigHandle, fillGeo, processSig, false);
	}
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);

    prepareParameters(rCutHandle, fillFCut);


    // 'true' has sendAllToDevice print after processing the parameters data.
    sendAllToDevice(true);
    setEvalWrapper();
    prepared = true;
    return prepared;
}

void FixLJCutFS::setEvalWrapper() {
    if (evalWrapperMode == "offload") {
        EvaluatorLJFS eval;
        evalWrap = pickEvaluator<EvaluatorLJFS, 3, true>(eval, chargeCalcFix);
    } else if (evalWrapperMode == "self") {
        EvaluatorLJFS eval;
        evalWrap = pickEvaluator<EvaluatorLJFS, 3, true>(eval, nullptr);
    } else {
        std::cout << "evalWrapperMode: " << evalWrapperMode << std::endl;
        std::cout << "evalWrapperMode does not correspond to either offload or self; aborting" << std::endl;
        assert(false);
    }
}

std::string FixLJCutFS::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}

void FixLJCutFS::printParams() {
    std::cout << "in FixLJCutFS::printParams()!" << std::endl;
    for (std::size_t i = 0; i < epsilons.size(); i++) {
        std::cout << "epsilons[" << i << "]: " << epsilons[i] << std::endl;
    }
    for (std::size_t i = 0; i < sigmas.size(); i++) {
        std::cout << "sigmas[" << i << "]: " << sigmas[i] << std::endl;
    }

    for (std::size_t i = 0; i < rCuts.size(); i++) {
        std::cout << "rCuts["  << i << "]: " << rCuts[i] << std::endl;
    }


}

// DEPRECATED
void FixLJCutFS::addSpecies(std::string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    initializeParameters("FCutHandle", FCuts);

}

std::vector<real> FixLJCutFS::getRCuts() {
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

void export_FixLJCutFS() {
    py::class_<FixLJCutFS,
                          boost::shared_ptr<FixLJCutFS>,
                          py::bases<FixPair>, boost::noncopyable > (
        "FixLJCutFS",
        py::init<boost::shared_ptr<State>, std::string, py::optional<std::string> > (
            py::args("state", "handle", "mixingRules"))
    )
    .def("printParams", &FixLJCutFS::printParams) 
    ;

}
