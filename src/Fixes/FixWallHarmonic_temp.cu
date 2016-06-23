#include "FixWallHarmonic_temp.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"

const std::string wallHarmonicType = "WallHarmonic";

// FixWallHarmonic_temp.cu
// need a constructor up in here (its a .cu!)



// this refers to the template in the /Evaluators/ folder - 
// will need a template, and an implementation for Harmonic walls as well
void FixWallHarmonic_temp::compute(bool computeVirials) {
	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();
	if (computeVirials) {
		// I think we just need the evaluator and whether or not to compute the virials - correct? we'll see..
		compute_wall_iso<EvaluatorWallHarmonic, true> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), origin.asFloat3(), forceDir.asFloat3(), dist, groupTag, evaluator);
	} else {
		compute_wall_iso<EvaluatorWallHarmonic, false> <<<NBLOCK(n), PERBLOCK>>>(n, gpd.xs(activeIdx),
                    gpd.fs(activeIdx), origin.asFloat3(), forceDir.asFloat3(), dist, groupTag, evaluator);
	}
};

void FixWallHarmonic_temp::singlePointEng(float *perParticleEng) {
	int nAtoms = state->atoms.size();

	//TODO is this needed for a wall? I would imagine, but the original harmonic wall doesnt have it implemented
};



bool prepareForRun() {
	return true;
};

bool postRun () {
	return true;
};








// export function
void export_WallHarmonic_temp() {
	py::class_<FixWallHarmonic_temp, boost::shared_ptr<FixWallHarmonic_temp>, py::bases<FixWall>, boost::noncopyable > (
		"FixWallHarmonic_temp",
		py::init<boost::shared_ptr<State>, string, string, Vector, Vector, double, double> (
			py::args("state", "handle", "groupHandle", "origin", "forceDir", "dist", "k")
		)
	) // it's expected that this will throw an error, via incorrectly accessing non-static members of the parent class?
	.def_readwrite("k", &FixWallHarmonic_temp::k)
	.def_readwrite("dist", &FixWallHarmonic_temp::dist)
	.def_readwrite("forceDir", &FixWallHarmonic_temp::forceDir)
	.def_readwrite("origin", &FixWallHarmonic_temp::origin)
	;
}


