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
FixDPD_T::FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
          double gamma_, double rcut_, double s_, int64_t seed_, py::list intervals_,
          py::list temps_) : 
          FixDPD(state_, handle_, groupHandle_, DPD_Type, intervals_, temps_), 
          gamma(gamma_), rcut(rcut_), s(s_), seed(seed_) {

             // the user provided gamma by construction, so we will hold it fixed
             // and update sigma instead when the temperature is varied
              updateGamma = false;
};
    
// constructor: given gamma, and a temperature function for the interpolator
FixDPD_T::FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
                  double gamma_, double rcut_, double s_, int64_t seed_, py::object tempFunc_) : 
          FixDPD(state_, handle_, groupHandle_, DPD_Type, tempFunc_), 
          gamma(gamma_), rcut(rcut_), s(s_), seed(seed_)
{ 
              updateGamma = false;
};

    
// constructor: given gamma, and a constant value for temperature for the interpolator
FixDPD_T::FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
                  double gamma_, double rcut_, double s_, int64_t seed_, double temp_) : 
          FixDPD(state_, handle_, groupHandle_, DPD_Type, temp_), 
          gamma(gamma_), rcut(rcut_), s(s_), seed(seed_) 
{ 

              updateGamma = false;
};
//
//// constructor: given sigma, and a list of interpolator intervals & temperatures
//FixDPD_T::FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
//                  double sigma_, double rcut_, double s_, int64_t seed_,  py::list intervals_,
//                  py::list temps_) : 
//          FixDPD(state_, handle_, groupHandle_, DPD_Type, intervals_, temps_), 
//          sigma(sigma_), rcut(rcut_), s(s_), seed(seed_)
//{
//    updateGamma = true;
//};
//
//// constructor: given sigma, and a temperature function for the interpolator
//FixDPD_T::FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
//                  double sigma_, double rcut_, double s_, int64_t seed_,  py::object tempFunc_) : 
//          FixDPD(state_, handle_, groupHandle_, DPD_Type, tempFunc_), 
//          sigma(sigma_), rcut(rcut_), s(s_), seed(seed_)
//{ 
//    updateGamma = true;
//};
//// constructor: given sigma, and a constant value for temperature for the interpolator
//FixDPD_T::FixDPD_T (boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
//                  double sigma_, double rcut_, double s_, int64_t seed_, double temp_):
//          FixDPD(state_, handle_, groupHandle_, DPD_Type,  temp_), 
//          sigma(sigma_), rcut(rcut_), s(s_), seed(seed_)
//{
//    updateGamma = true;
//};
//
void FixDPD_T::hashSeed(int64_t seed) {



};

void FixDPD_T::updateParameters(double currentTemperature) {
    if (Interpolator::mode != thermoType::constant) {
        if (updateGamma) {
            double newGamma = (evaluator.sigma ** 2.0) / (currentTemperature * 2.0 ) ;
            evaluator.updateGamma(newGamma);
        } else {
            double newSigma = sqrt(2.0 * currentTemperature * evaluator.gamma);
            evaluator.updateSigma(newSigma);
        }
    }
};


// our compute function
void FixDPD_T::compute(bool computeVirials) {
	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();
    GridGPU &grid = state->gridGPU;
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    int64_t turn = state->turn;
    computeCurrentVal(turn);
    double temp = getCurrentVal();
    updateParameters(temp);
    
    computeDPD_Isothermal<EvaluatorDPD_T, false> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), gpd.vs(activeIdx),  gpd.ids(activeIdx), gpd.fds(activeIdx),
                    neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                    state->devManager.prop.warpSize, state->turn, state->boundsGPU,
                    groupTag,  evaluator);

};



void FixDPD_T::singlePointEng(float *perParticleEng) {

};

void FixDPD_T::stepFinal( ) {
    // update the dissipative forces for the initial integration step on next turn
	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();
    GridGPU &grid = state->gridGPU;
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    int64_t turn = state->turn;
    // the temperature should not have changed between now and compute(), I think
    // likewise, the parameters sigma and gamma should not have changed between now and compute
    // (since they are temperature dependent)
    // what about a global seed to be sent to SARU? should be here.
    computeDPD_Isothermal<EvaluatorDPD_T, true> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), gpd.vs(activeIdx),  gpd.ids(activeIdx), gpd.fds(activeIdx),
                    neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                    state->devManager.prop.warpSize, state->turn, state->boundsGPU,
                    groupTag,  evaluator);
};



bool FixDPD_T::prepareForRun() {
    // instantiate this fix's evaulator with the appropriate parameters
    // compute sigma (or gamma), pass this information to the evaluator
    // along with rcut, integer s

    // call base class Interpolator's functions to compute and retrieve the current temperature
    computeCurrentVal(turn);
    double temperature = getCurrentVal();

    // calculate sigma or gamma as needed; note that the evaluator has not yet been instantiated
    // 
    // consider a run where the temperature is gradually increased to a setpoint, the run stops,
    // some new fixes are added - and then the simulation is instructed to continue;
    // wouldn't we want it to continue from the previous point? the oldest value of sigma & gamma?
    // if so, we need to consider what goes in prepareForRun();
    // nah, in postRun() just set the values of sigma & gamma from the evaluator as the class variable values
    // that way we don't run in to this issue! crisis averted.
    if (updateGamma) {
       gamma = (sigma ** 2.0) / (2.0 * currentTemperature) ;
    } else {
       sigma = sqrt(2.0 * currentTemperature * gamma);
    }

    // we modify the random thermal fluctuations by inverse square root of the timestep
    // so that they have the same units as the other forces when aggregated in the fs[idx] array 
    double invsqrtdt = 1.0f / sqrt(state->dt);

    // hashes the seed provided by the user using a hash function
    hashSeed(seed);
    
    // also, hash the global seed that they provided in the constructor
    // instantiate the evaluator
    evaluator = EvaluatorDPD_T(sigma, gamma, rcut, s, invsqrtdt);

    // exit and return to where we were called (Integrator::prepareForRun()
    return true;
};


bool FixDPD_T::postRun () {
    gamma = evaluator.gamma;
    sigma = evaluator.sigma;
    return true;
};

// export function
void export_FixDPD_T() 
{ 
// export the various constructors - intervals -> function -> const temp
    py::class_<FixDPD_T, 
        SHARED(FixDPD_T), 
        py::bases<FixDPD> > 
            (
        "FixDPD_T", 
        py::init<boost::shared_ptr<State>, std::string, std::string, double, double, double, int64_t, py::list, py::list> (
            py::args("state", "handle", "groupHandle", "gamma", "rcut", "s", "seed", "intervals", "temps")
            )
        )
    .def(py::init<boost::shared_ptr<State>,std::string,std::string, double, double, double, int64_t, py::object> (
            py::args("state", "handle", "groupHandle", "gamma", "rcut", "s", "seed", "tempFunc")
            )
        )
    .def(py::init<boost::shared_ptr<State>,std::string,std::string, double, double, double, int64_t, double> (
            py::args("state", "handle", "groupHandle", "gamma", "rcut", "s", "seed", "temp")
            )
        );
    //// and the same, but now with sigma specified rather than gamma
    //.def(py::init<boost::shared_ptr<State>,std::string,std::string, double, double, double, int64_t, py::list, py::list> (
    //       py::args("state", "handle", "groupHandle", "sigma", "rcut", "s", "seed", "intervals", "temps")
    //       )
    //    )
    //.def(py::init<boost::shared_ptr<State>,std::string,std::string, double, double, double, int64_t, py::object> (
    //        py::args("state", "handle", "groupHandle", "sigma", "rcut", "s", "seed", "tempFunc")
    //        )
    //    )
    //.def(py::init<boost::shared_ptr<State>,std::string,std::string, double, double, double, int64_t, double> (
    //        py::args("state", "handle", "groupHandle", "sigma", "rcut", "s", "seed", "temp")
    //        )
    //    );
};



