#include "FixDPD_T.h"
#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"


const std::string DPD_Type = "isothermalDPD";
namespace py = boost::python;

// here, we place the implementations of the constructors

// our compute function
void FixDPD_T::compute(bool computeVirials) {
	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();

    // alter here for the proper evaluator
	if (computeVirials) {
		compute_DPD_iso<EvaluatorDPD_T, true> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), groupTag, evaluator);
	} else {
		compute_DPD_iso<EvaluatorDPD_T, false> <<<NBLOCK(n), PERBLOCK>>>(n, gpd.xs(activeIdx),
                    gpd.fs(activeIdx), groupTag, evaluator);
	}



};



void FixDPD_T::singlePointEng(float *perParticleEng) {

};

void FixDPD_T::stepFinal( ) {
    // update the dissipative forces for the initial integration step on next turn

    // do this by instantiating a kernel call
    if (computeVirials) {
         compute_DPD_iso<EvaluatorDPD_T, true> <<<NBLOCK(n), PERBLOCK>>> (n, gpd.xs(activeIdx),
                gpd.fs(activeIdx), groupTag, evaluator);
    } else {
         compute_DPD_iso<EvaluatorDPD_T, false> <<<NBLOCK(n), PERBLOCK>>> (n, gpd.xs(activeIdx),
                gpd.fs(activeIdx), groupTag, evaluator);
    };
};



bool FixDPD_T::prepareForRun() {
    // instantiate this fix's evaulator with the appropriate parameters
    //evaluator = EvaluatorDPD_T();

    return true;
};


bool FixDPD_T::postRun () {
    return true;
};

// export function
void export_FixDPD_T() {

};


