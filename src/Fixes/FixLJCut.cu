#include "FixLJCut.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"
const std::string LJCutType = "LJCut";

__device__ void eval (float3 &forceSum, float3 dr, float *params, float lenSqr, float multiplier) {
    printf("here\n");
    float rCutSqr = params[2];
    if (lenSqr < rCutSqr) {
        float epstimes24 = params[0];
        float sig6 = params[1];
        float p1 = epstimes24*2*sig6*sig6;
        float p2 = epstimes24*sig6;
        float r2inv = 1/lenSqr;
        float r6inv = r2inv*r2inv*r2inv;
        float forceScalar = r6inv * r2inv * (p1 * r6inv - p2) * multiplier;

        float3 forceVec = dr * forceScalar;
        forceSum += forceVec;
    }
}


FixLJCut::FixLJCut(boost::shared_ptr<State> state_, std::string handle_)
  : FixPair(state_, handle_, "all", LJCutType, true, 1),
    epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut")
{
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {epsHandle, sigHandle, rCutHandle};
}

void FixLJCut::compute(bool computeVirials) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

   SAFECALL(( compute_force_iso<EvaluatorLJ, 3> <<<NBLOCK(nAtoms), PERBLOCK, 3*numTypes*numTypes*sizeof(float)>>>(
            nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
            neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
            state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
            neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], evaluator)
            ));

}

void FixLJCut::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    compute_energy_iso<EvaluatorLJ, 3><<<NBLOCK(nAtoms), PERBLOCK, 3*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), perParticleEng, 
                                                                                                        neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], evaluator);



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

std::string FixLJCut::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}

bool FixLJCut::readFromRestart(pugi::xml_node restData) {
    std::cout << "Reading form restart" << std::endl;
    auto curr_param = restData.first_child();
    while (curr_param) {
        if (curr_param.name() == "parameter") {
            std::vector<float> val;
            std::string paramHandle = curr_param.attribute("handle").value();
            std::string s;
            std::istringstream ss(curr_param.value());
            while (ss >> s) {
                val.push_back(atof(s.c_str()));
            }
            initializeParameters(paramHandle, val);
        }
        curr_param = curr_param.next_sibling();
    }
    std::cout << "Reading LJ parameters from restart" << std::endl;
    return true;
}

bool FixLJCut::postRun() {
    resetToPreproc(sigHandle);
    resetToPreproc(epsHandle);
    resetToPreproc(rCutHandle);

    return true;
}

void FixLJCut::addSpecies(std::string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);

}

std::vector<float> FixLJCut::getRCuts() {  // to be called after prepare.  These are squares now
    return LISTMAP(float, float, rc, rCuts, sqrt(rc));
}

void export_FixLJCut() {
    boost::python::class_<FixLJCut,
                          boost::shared_ptr<FixLJCut>,
                          boost::python::bases<FixPair>, boost::noncopyable > (
        "FixLJCut",
        boost::python::init<boost::shared_ptr<State>, std::string> (
            boost::python::args("state", "handle"))
    );

}
