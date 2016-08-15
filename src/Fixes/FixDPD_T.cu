#include "FixDPD_T.h"
#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"


const std::string DPD_Type = "isothermalDPD";
namespace py = boost::python;

// here, we place the implementations of the constructors

void FixDPD_T::compute(bool computeVirials) {



};



void FixDPD_T::singlePointEng(float *perParticleEng) {

};


bool FixDPD_T::prepareForRun() {
    // instantiate this fix's evaulator with the appropriate parameters
    // evaluator = EvaluatorDPD_T();

    return true;
};


bool FixDPD_T::postRun () {
    return true;
};

// export function
void export_FixDPD_T() {

};


