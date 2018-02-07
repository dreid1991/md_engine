#include "FixWCA.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"
#include "EvaluatorWrapper.h"

const std::string LJCutType = "LJCutWCA";
namespace py = boost::python;

FixWCA::FixWCA(SHARED(State) state_, std::string handle_, std::string mixingRules_)
    : FixPair(state_, handle_, "all", LJCutType, true, false, 1, ARITHMETICTYPE),
      epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut") {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {rCutHandle, epsHandle, sigHandle};
    readFromRestart();
    setEvalWrapper();
}
void FixWCA::compute(int virialMode) {
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

void FixWCA::singlePointEng(real *perParticleEng) {
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

bool FixWCA::prepareForRun() {
    //loop through all params and fill with appropriate lambda function, then send all to device
    auto fillGeo = [] (real a, real b) {
        return sqrt(a*b);
    };

    auto fillArith = [] (real a, real b) {
        return (a+b) / 2.0;
    };
//     auto fillRCut = [this] (real a, real b) {
//         return (real) std::fmax(a, b);
//     };
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

    auto fillRCut = [this] (int a, int b) {
        int numTypes = state->atomParams.numTypes;
        real sig = squareVectorRef<real>(paramMap[sigHandle]->data(),numTypes,a,b);
        return sig*pow(2.0,1.0/6.0);
    };    
    prepareParameters(epsHandle, fillGeo, processEps, false);
	if (mixingRules=="arithmetic") {
		prepareParameters(sigHandle, fillArith, processSig, false);
	} else {
		prepareParameters(sigHandle, fillGeo, processSig, false);
	}
    prepareParameters_from_other(rCutHandle, fillRCut, processRCut, false);

//     prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);
    sendAllToDevice();
    setEvalWrapper();
    prepared = true;
    return prepared;
}

void FixWCA::setEvalWrapper() {
    if (evalWrapperMode == "offload") {
        EvaluatorWCA eval;
        evalWrap = pickEvaluator<EvaluatorWCA, 3, true>(eval, chargeCalcFix);
    } else if (evalWrapperMode == "self") {
        EvaluatorWCA eval;
        evalWrap = pickEvaluator<EvaluatorWCA, 3, true>(eval, nullptr);
    }

}

std::string FixWCA::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}



void FixWCA::addSpecies(std::string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);

}

std::vector<real> FixWCA::getRCuts() {
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

bool FixWCA::setParameter(std::string param,
                           std::string handleA,
                           std::string handleB,
                           double val)
{
      if (param==sigHandle) FixPair::setParameter(rCutHandle, handleA,handleB,val*pow(2.0,1.0/6.0));
      return FixPair::setParameter(param, handleA,handleB,val);
      
}

void export_FixWCA() {
    py::class_<FixWCA,
                          SHARED(FixWCA),
                          py::bases<FixPair>, boost::noncopyable > (
        "FixWCA",
        py::init<SHARED(State), std::string, py::optional<std::string> > (
            py::args("state", "handle", "mixingRules")))
        .def("setParameter", &FixWCA::setParameter,
                ( py::arg("param"),
                  py::arg("handleA"),
                  py::arg("handleB"),
                  py::arg("val"))
            )
        ;

}

/*
void export_FixWCA() {
    py::class_<FixWCA,
                          SHARED(FixWCA),
                          py::bases<FixPair>, boost::noncopyable > (
        "FixWCA",
        py::init<SHARED(State), std::string, py::optional<std::string> > (
            py::args("state", "handle", "mixingRules")))
        .def("setParameter", &FixWCA::setParameter,
                ( py::arg("param"),
                  py::arg("handleA"),
                  py::arg("handleB"),
                  py::arg("val"))
            )
        ;

}
*/
