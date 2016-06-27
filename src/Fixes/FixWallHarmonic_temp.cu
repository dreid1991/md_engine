#include "FixWallHarmonic_temp.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "WallEvaluate.h"

const std::string wallHarmonicType = "WallHarmonic_temp";
using namespace std;
namespace py = boost::python;

// the constructor for FixWallHarmonic
FixWallHarmonic_temp::FixWallHarmonic_temp(SHARED(State) state_, std::string handle_, std::string groupHandle_,
                                 Vector origin_, Vector forceDir_, float dist_, float k_)
  : FixWall(state_, handle_, groupHandle_, wallHarmonicType, true,  false, 1, origin_, forceDir_.normalized()),
    dist(dist_), k(k_)
{
    assert(dist >= 0);
};



// this refers to the template in the /Evaluators/ folder - 
// will need a template, and an implementation for Harmonic walls as well
void FixWallHarmonic_temp::compute(bool computeVirials) {
	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();
	if (computeVirials) {
		// I think we just need the evaluator and whether or not to compute the virials - correct? we'll see..
		// ^ referring to what to pass in as template specifiers
		compute_wall_iso<EvaluatorWallHarmonic, true> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), origin.asFloat3(), forceDir.asFloat3(), dist, groupTag, evaluator);
	} else {
		compute_wall_iso<EvaluatorWallHarmonic, false> <<<NBLOCK(n), PERBLOCK>>>(n, gpd.xs(activeIdx),
                    gpd.fs(activeIdx), origin.asFloat3(), forceDir.asFloat3(), dist, groupTag, evaluator);
	}
};

void FixWallHarmonic_temp::singlePointEng(float *perParticleEng) {

};



bool FixWallHarmonic_temp::prepareForRun() {
    // instantiate this fix's evaulator with the appropriate parameters

    evaluator.setParameters(k,dist);
    return true;
};

bool FixWallHarmonic_temp::postRun () {
    return true;
};








// export function
// and this is where all the errors are now :) to fix!
void export_FixWallHarmonic_temp() {
	py::class_<FixWallHarmonic_temp, SHARED(FixWallHarmonic_temp), py::bases<FixWall>, boost::noncopyable > (
		"FixWallHarmonic_temp",
		py::init<SHARED(State), string, string, Vector, Vector, float, float> (
			py::args("state", "handle", "groupHandle", "origin", "forceDir", "dist", "k")
		)
	)
	.def_readwrite("k", &FixWallHarmonic_temp::k)
	.def_readwrite("dist", &FixWallHarmonic_temp::dist)
	.def_readwrite("forceDir", &FixWallHarmonic_temp::forceDir)
	.def_readwrite("origin", &FixWallHarmonic_temp::origin)
	;
}


