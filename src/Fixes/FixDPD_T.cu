#include "FixDPD_T.h"
#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"


const std::string DPD_Type = "isothermalDPD";
namespace py = boost::python;

// here, we place the implementations of the constructors

// constructor: given gamma, and a list of interpolator intervals & temperatures
FixDPD_T::FixDPD_T (State* state_, std::string handle_, std::string groupHandle_,
          double gamma_, double rcut_, double s_, py::list intervals_,
          py::list temps_) : 
          FixDPD(state_, handle_, groupHandle_, DPD_Type, intervals_, temps_), 
          gamma(gamma_), rcut(rcut_), s(s_) {

             // the user provided gamma by construction, so we will hold it fixed
             // and update sigma instead when the temperature is varied
              updateGamma = false;
};
    
// constructor: given gamma, and a temperature function for the interpolator
FixDPD_T::FixDPD_T (State* state_, std::string handle_, std::string groupHandle_,
                  double gamma_, double rcut_, double s_, py::object tempFunc_) : 
          FixDPD(state_, handle_, groupHandle_, DPD_Type, tempFunc_), 
          gamma(gamma_), rcut(rcut_), s(s_) 
{ 
              updateGamma = false;
};

    
// constructor: given gamma, and a constant value for temperature for the interpolator
FixDPD_T::FixDPD_T (State* state_, std::string handle_, std::string groupHandle_,
                  double gamma_, double rcut_, double s_, double temp_) : 
          FixDPD(state_, handle_, groupHandle_, DPD_Type, temp_), 
          gamma(gamma_), rcut(rcut_), s(s_) 
{ 

              updateGamma = false;
};
// constructor: given sigma, and a list of interpolator intervals & temperatures
FixDPD_T::FixDPD_T (State* state_, std::string handle_, std::string groupHandle_,
                  double sigma_, double rcut_, double s_, py::list intervals_,
                  py::list temps_) : 
          FixDPD(state_, handle_, groupHandle_, DPD_Type, intervals_, temps_), 
          sigma(sigma_), rcut(rcut_), s(s_) 
{
    updateGamma = true;
};

// constructor: given sigma, and a temperature function for the interpolator
FixDPD_T::FixDPD_T (State* state_, std::string handle_, std::string groupHandle_,
                  double sigma_, double rcut_, double s_, py::object tempFunc_) : 
          FixDPD(state_, handle_, groupHandle_, DPD_Type, tempFunc_), 
          sigma(sigma_), rcut(rcut_), s(s_) 
{ 
    updateGamma = true;
};
// constructor: given sigma, and a constant value for temperature for the interpolator
FixDPD_T::FixDPD_T (State* state_, std::string handle_, std::string groupHandle_,
                  double sigma_, double rcut_, double s_, double temp_):
          FixDPD(state_, handle_, groupHandle_, DPD_Type,  temp_), 
          sigma(sigma_), rcut(rcut_), s(s_)
{
    updateGamma = true;
};

// our compute function
void FixDPD_T::compute(bool computeVirials) {
	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();
    GridGPU &grid = state->gridGPU;
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    
    
    computeDPD_Isothermal<EvaluatorDPD_T> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), gpd.vs(activeIdx),  gpd.ids(activeIdx), gpd.fds(activeIdx),
                    neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                    state->devManager.prop.warpSize, state->turn, state->boundsGPU,
                    groupTag,  evaluator);

};



void FixDPD_T::singlePointEng(float *perParticleEng) {

};

void FixDPD_T::stepFinal( ) {
    // update the dissipative forces for the initial integration step on next turn

    // do this by instantiating a kernel call
    // correct this kernel call, it is very not correct at the moment
    compute_DPD_Isothermal_StepFinal<EvaluatorDPD_T> <<<NBLOCK(n), PERBLOCK>>> (n, gpd.xs(activeIdx),
                gpd.fs(activeIdx), groupTag, evaluator);
};



bool FixDPD_T::prepareForRun() {
    // instantiate this fix's evaulator with the appropriate parameters
    // compute sigma (or gamma), pass this information to the evaluator
    // along with rcut, integer s
    double temperature = state->Interpolator.getCurrentVal();
    evaluator = EvaluatorDPD_T();
    // fetch sqrtdt, compute either gamma or sigma as needed

    // where will we be putting the updateGamma() or updateSigma() method?
    return true;
};


bool FixDPD_T::postRun () {
    return true;
};

// export function
void export_FixDPD_T() {

};


