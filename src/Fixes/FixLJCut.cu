#include "FixLJCut.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"
using namespace std;
namespace py = boost::python;
const string LJCutType = "LJCut";



FixLJCut::FixLJCut(boost::shared_ptr<State> state_, string handle_)
  : FixPair(state_, handle_, "all", LJCutType, true, false, 1),
    epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut")
{
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {rCutHandle, epsHandle, sigHandle};
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

string FixLJCut::restartChunk(string format) {
    stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}

bool FixLJCut::readFromRestart(pugi::xml_node restData) {
    cout << "Reading form restart" << endl;
    auto curr_param = restData.first_child();
    while (curr_param) {
        if (curr_param.name() == "parameter") {
            vector<float> val;
            string paramHandle = curr_param.attribute("handle").value();
            string s;
            istringstream ss(curr_param.value());
            while (ss >> s) {
                val.push_back(atof(s.c_str()));
            }
            initializeParameters(paramHandle, val);
        }
        curr_param = curr_param.next_sibling();
    }
    cout << "Reading LJ parameters from restart" << endl;
    return true;
}

bool FixLJCut::postRun() {

    return true;
}

void FixLJCut::addSpecies(string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);

}

vector<float> FixLJCut::getRCuts() { 
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

void export_FixLJCut() {
    py::class_<FixLJCut, boost::shared_ptr<FixLJCut>, py::bases<FixPair>, boost::noncopyable > (
        "FixLJCut",
        py::init<boost::shared_ptr<State>, string> (py::args("state", "handle"))
    );

}
