
#include "FixDeform.h"
#include "Mod.h"
#include "Interpolator.h"
#include "State.h"
//particles in group handle given will be scaled with the box on deformation
//all and none group handles will scale all or none of the atoms

using std::cout;
using std::endl;

std::string DeformType = "Deform";
FixDeform::FixDeform(boost::shared_ptr<State> state_, std::string handle_,
        std::string groupHandle_, double deformRate_, Vector multiplier_, int applyEvery_) : Fix(state_, handle_, groupHandle_, DeformType, false, false, false, applyEvery_), deformRateInterpolator(deformRate_), multiplier(multiplier_) {
}

FixDeform::FixDeform(boost::shared_ptr<State> state_, std::string handle_,
        std::string groupHandle_, py::object deformRateFunc_, Vector multiplier_, int applyEvery_) : Fix(state_, handle_, groupHandle_, DeformType, false, false, false, applyEvery_), deformRateInterpolator(deformRateFunc_), multiplier(multiplier_)  {
}

FixDeform::FixDeform(boost::shared_ptr<State> state_, std::string handle_,
        std::string groupHandle_, py::list intervals_, py::list rates_, Vector multiplier_, int applyEvery_)  : Fix(state_, handle_, groupHandle_, DeformType, false, false, false, applyEvery_), deformRateInterpolator(intervals_, rates_), multiplier(multiplier_)  {
}


bool FixDeform::prepareForRun() {
    deformRateInterpolator.turnBeginRun = state->runInit;
    deformRateInterpolator.turnFinishRun = state->runInit + state->runningFor;
    return true;
}

bool FixDeform::stepFinal() {
    deformRateInterpolator.computeCurrentVal(state->turn);
    double rate = deformRateInterpolator.getCurrentVal();
    float3 deltaBounds = (multiplier * rate * state->dt).asFloat3();
    float3 newTrace = state->boundsGPU.rectComponents + deltaBounds;
    float3 scaleBy = newTrace / state->boundsGPU.rectComponents;
    Mod::scaleSystem(state, scaleBy, groupTag);
    return true;

}


void export_FixDeform()
{
    py::class_<FixDeform,                    // Class
               boost::shared_ptr<FixDeform>, // HeldType
               py::bases<Fix>,                   // Base class
               boost::noncopyable>
    (
        "FixDeform",
        py::init<boost::shared_ptr<State>, std::string, std::string, py::object, py::optional<Vector, int> >(
            py::args("state", "handle", "groupHandle", "deformFunc", "multiplier", "applyEvery")
        )
    )
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, py::list, py::list, py::optional<Vector, int> >(
                py::args("state", "handle", "groupHandle", "intervals", "deformRates", "multiplier", "applyEvery")

                )
        )
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, double, py::optional<Vector, int> >(
                py::args("state", "handle", "groupHandle", "deformRate", "multiplier", "applyEvery")

                )
        )

    ;
}

