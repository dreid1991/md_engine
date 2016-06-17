#include "FixLJCut.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"
#include "ReadConfig.h"

const std::string LJCutType = "LJCut";

FixLJCut::FixLJCut(boost::shared_ptr<State> state_, std::string handle_)
  : FixPair(state_, handle_, "all", LJCutType, true, 1),
    epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut")
{
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {epsHandle, sigHandle, rCutHandle};
    
    if (state->readConfig->fileOpen) {
      auto restData = state->readConfig->readFix(type, handle);
      if (restData) {
	std::cout << "Reading restart data for fix " << handle << std::endl;
	readFromRestart(restData);
      }
    }
}

void FixLJCut::compute(bool computeVirials) {

    int nAtoms = state->atoms.size();
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    compute_force_iso<EvaluatorLJ, 3> <<<NBLOCK(nAtoms), PERBLOCK, 3*numTypes*numTypes*sizeof(float)>>>(
            nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx),
            neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
            state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
            neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], evaluator);

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

std::string FixLJCut::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}

bool FixLJCut::readFromRestart(pugi::xml_node restData) {
    std::cout << "Reading form restart" << std::endl;
    auto curr_param = restData.first_child();
    while (curr_param) {
      std::string tag = curr_param.name();
        if (tag == "parameter") {
            std::vector<float> val;
            std::string paramHandle = curr_param.attribute("handle").value();
            std::string s;
            std::istringstream ss(curr_param.value());
            while (ss >> s) {
                val.push_back(atof(s.c_str()));
	    }
	    if (paramHandle == epsHandle) {
	      epsilons = val;
	    } else if (paramHandle == sigHandle) {
	      sigmas = val;
	    } else if (paramHandle == rCutHandle) {
	      rCuts = val;
	    } else {
	      std::cout << "Error: Invalid FixLJCut parameter handle, " << paramHandle << std::endl;
	      return false;
	    }
        }
        curr_param = curr_param.next_sibling();
    }
    int count = 0;
    for (int i = 0; i < state->atomParams.handles.size(); i++) {
      for (int j = 0; j < state->atomParams.handles.size(); j++) {
	std::cout << "setting parameters" << std::endl;
	  setParameter(epsHandle, state->atomParams.handles[i], state->atomParams.handles[j], epsilons[count]);
	  setParameter(sigHandle, state->atomParams.handles[i], state->atomParams.handles[j], sigmas[count]);
	  setParameter(rCutHandle, state->atomParams.handles[i], state->atomParams.handles[j], rCuts[count]);
	  count++;
      }
    }
    std::cout << "Read LJ parameters from restart" << std::endl;
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
    )
      .def_readonly("epsilons", &FixLJCut::epsilons)
      .def("restartChunk", &FixLJCut::restartChunk,
	   (boost::python::arg("format")="xml"))
      ;
}
