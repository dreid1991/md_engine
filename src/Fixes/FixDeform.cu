
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
        std::string groupHandle_, double deformRate_, Vector multiplier_, int applyEvery_) : Fix(state_, handle_, groupHandle_, DeformType, false, false, false, applyEvery_), multiplier(multiplier_) {
    setPtVolume = -1;
}

FixDeform::FixDeform(boost::shared_ptr<State> state_, std::string handle_,
        std::string groupHandle_, py::object deformRateFunc_, Vector multiplier_, int applyEvery_) : Fix(state_, handle_, groupHandle_, DeformType, false, false, false, applyEvery_), multiplier(multiplier_)  {
    setPtVolume = -1;
}

FixDeform::FixDeform(boost::shared_ptr<State> state_, std::string handle_,
        std::string groupHandle_, py::list intervals_, py::list rates_, Vector multiplier_, int applyEvery_)  : Fix(state_, handle_, groupHandle_, DeformType, false, false, false, applyEvery_), multiplier(multiplier_)  {
    setPtVolume = -1;
}


bool FixDeform::prepareForRun() {

	/*
	massTotal = 0;
	for (Atom &a : state->atoms) {
		massTotal += a.mass;
	}
	*/

    if (setPtVolume != -1) {
        //then override entered rate
        double curVol = state->bounds.volume();
        double volRatio = setPtVolume / curVol;
		//printf("WANT VOL %f\n", curVol * volRatio);

		//double density = (massTotal / (curVol * volRatio)) * state->units.toSIDensity;
		//printf("WANT DENSITY %f\n", density);
        int nDimDeform = 0;
        for (int i=0; i<3; i++) {
            nDimDeform += multiplier[i]>0 ? 1 : 0;
        }
        double sideLenRatio = pow(volRatio, 1.0/nDimDeform);
        for (int i=0; i<3; i++) {
            if (multiplier[i]>0) {
                multiplier[i] = (state->bounds.rectComponents[i]) / state->bounds.rectComponents[0];
            }
        }
        rate = -state->bounds.rectComponents[0] * (1-sideLenRatio) / (state->runningFor);

        

    }
	traceBeginRun = Vector(state->bounds.rectComponents);
    prepared = true;
    return prepared;
}

bool FixDeform::stepFinal() {
	//okay, to avoid numerical error, the deform fix is going to set the side length based on calculated rate and # turns elapsed
	
	//double density = (massTotal / state->boundsGPU.volume()) * state->units.toSIDensity;
	//printf("DENSITY %f\n", density);


	//Vector fin = traceBeginRun + multiplier * rate * state->runningFor;
	//density = (massTotal / (fin[0]*fin[1]*fin[2])) * state->units.toSIDensity;

	//printf("DENSITY CHEAT %f\n", density);
	

	Vector totalDeform = multiplier * rate * (state->turn - state->runInit);
	Vector trace = traceBeginRun + totalDeform;
	Vector cur = Vector(state->boundsGPU.rectComponents);
	Vector ratio = trace / cur;
	float3 asFloat3 = ratio.asFloat3();
	//std::cout << totalDeform << std::endl;	
	Mod::scaleSystem(state, asFloat3, groupTag);

	/*
    deformRateInterpolator.computeCurrentVal(state->turn);
    double rate = deformRateInterpolator.getCurrentVal();
    real3 deltaBounds = (multiplier * rate * state->dt).asreal3();
    real3 newTrace = state->boundsGPU.rectComponents + deltaBounds;
    real3 scaleBy = newTrace / state->boundsGPU.rectComponents;
    Mod::scaleSystem(state, scaleBy, groupTag);
	*/
    return true;

}

void FixDeform::toVolume(double volume) {
    setPtVolume = volume;
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
    .def("toVolume", &FixDeform::toVolume)

    ;
}

