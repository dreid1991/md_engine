#include "FixLJCutFS.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"

const std::string LJCutType = "LJCutFS";

FixLJCutFS::FixLJCutFS(SHARED(State) state_, std::string handle_)
    : FixPair(state_, handle_, "all", LJCutType, true, false, 1),
      epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut") {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    initializeParameters("FCutHandle", FCuts);
    paramOrder = {rCutHandle, epsHandle, sigHandle, "FCutHandle"};
}
void FixLJCutFS::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    if (computeVirials) {
        compute_force_iso<EvaluatorLJFS, 4, true>  <<<NBLOCK(nAtoms), PERBLOCK, 4*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), 
                neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, 
                neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), evaluator);
    } else {
        compute_force_iso<EvaluatorLJFS, 4, false>  <<<NBLOCK(nAtoms), PERBLOCK, 4*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), 
                neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, 
                neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), evaluator);
    }



}

void FixLJCutFS::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    compute_energy_iso<EvaluatorLJFS, 4><<<NBLOCK(nAtoms), PERBLOCK, 4*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), perParticleEng, neighbor\
Counts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, ne\
ighborCoefs[0], neighborCoefs[1], neighborCoefs[2], evaluator);



}

bool FixLJCutFS::prepareForRun() {
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
        return (float) state->rCut;//WHY??
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
    
    auto fillFCut = [this] (int a, int b) {
        int numTypes = state->atomParams.numTypes;
        float epstimes24=24*squareVectorRef<float>(paramMap[epsHandle]->data(),numTypes,a,b);
        float rCutSqr = pow(squareVectorRef<float>(paramMap[rCutHandle]->data(),numTypes,a,b),2);
        float sig6 = pow(squareVectorRef<float>(paramMap[sigHandle]->data(),numTypes,a,b),6);
        float p1 = epstimes24*2*sig6*sig6;
        float p2 = epstimes24*sig6;
        float r2inv = 1/rCutSqr;
        float r6inv = r2inv*r2inv*r2inv;
        float forceScalar = r6inv * r2inv * (p1 * r6inv - p2)*sqrt(rCutSqr);

        return forceScalar;
    };
    prepareParameters(epsHandle, fillEps, processEps, false);
    prepareParameters(sigHandle, fillSig, processSig, false);
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);
    prepareParameters("FCutHandle", fillFCut);
    sendAllToDevice();
    return true;
}

std::string FixLJCutFS::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}


bool FixLJCutFS::postRun() {

    return true;
}

void FixLJCutFS::addSpecies(std::string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    initializeParameters(rCutHandle, FCuts);

}

std::vector<float> FixLJCutFS::getRCuts() {
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

void export_FixLJCutFS() {
    boost::python::class_<FixLJCutFS,
                          SHARED(FixLJCutFS),
                          boost::python::bases<FixPair>, boost::noncopyable > (
        "FixLJCutFS",
        boost::python::init<SHARED(State), std::string> (
            boost::python::args("state", "handle"))
    );

}
