#include "FixTICG.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"

const std::string TICGType = "TICG";

FixTICG::FixTICG(boost::shared_ptr<State> state_, std::string handle_)
  : FixPair(state_, handle_, "all", TICGType, true, false, 1),
    CHandle("C"),  rCutHandle("rCut")
{
    initializeParameters(CHandle, Cs);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {rCutHandle, CHandle};
}

void FixTICG::compute(bool computeVirials) {

    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    compute_force_iso<EvaluatorTICG, 2> <<<NBLOCK(nAtoms), PERBLOCK, 2*numTypes*numTypes*sizeof(float)>>>(
            nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
            neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
            state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
            neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], evaluator);

}

void FixTICG::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    compute_energy_iso<EvaluatorTICG, 2><<<NBLOCK(nAtoms), PERBLOCK, 2*numTypes*numTypes*sizeof(float)>>>(nAtoms, gpd.xs(activeIdx), perParticleEng, neighbor\
Counts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, ne\
ighborCoefs[0], neighborCoefs[1], neighborCoefs[2], evaluator);



}

bool FixTICG::prepareForRun() {
    //loop through all params and fill with appropriate lambda function, then send all to device
    auto none = [] (float a){};

    auto processCs = [] (float a) {
        return a;
    };
    auto processRCut = [] (float a) {
        return a*a;
    };
    prepareParameters(CHandle,  processCs);
    prepareParameters(rCutHandle, processRCut);
    sendAllToDevice();
    return true;
}

std::string FixTICG::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}

bool FixTICG::readFromRestart(pugi::xml_node restData) {
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
    std::cout << "Reading TICG parameters from restart" << std::endl;
    return true;
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
