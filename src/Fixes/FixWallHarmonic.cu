#include "FixWallHarmonic.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "WallEvaluate.h"

const std::string wallHarmonicType = "WallHarmonic";
namespace py = boost::python;

// the constructor for FixWallHarmonic
FixWallHarmonic::FixWallHarmonic(SHARED(State) state_, std::string handle_, std::string groupHandle_,
                                 Vector origin_, Vector forceDir_, real dist_, real k_)
  : FixWall(state_, handle_, groupHandle_, wallHarmonicType, true,  false, 1, origin_, forceDir_.normalized()),
    dist(dist_), k(k_)
{
    assert(dist >= 0);
};



// this refers to the template in the /Evaluators/ folder - 
// will need a template, and an implementation for Harmonic walls as well
void FixWallHarmonic::compute(int virialMode) {
	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();
	if (virialMode) {
		// I think we just need the evaluator and whether or not to compute the virials - correct? we'll see..
		// ^ referring to what to pass in as template specifiers
		compute_wall_iso<EvaluatorWallHarmonic, true> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), 
                    gpd.virials.d_data.data(),
                    origin.asreal3(), 
                    forceDir.asreal3(),  
                    groupTag, 
                    evaluator);
	} else {
		compute_wall_iso<EvaluatorWallHarmonic, false> <<<NBLOCK(n), PERBLOCK>>>(n, gpd.xs(activeIdx),
                    gpd.fs(activeIdx), 
                    gpd.virials.d_data.data(),
                    origin.asreal3(), 
                    forceDir.asreal3(),  
                    groupTag,
                    evaluator);
	}
};

void FixWallHarmonic::singlePointEng(real *perParticleEng) {

	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();

    compute_wall_energy<EvaluatorWallHarmonic> <<<NBLOCK(n), PERBLOCK>>>(n,
                                                                         gpd.xs(activeIdx),
                                                                         perParticleEng,
                                                                         gpd.fs(activeIdx), // for groupTag
                                                                         origin.asreal3(),
                                                                         forceDir.asreal3(),
                                                                         groupTag,
                                                                         evaluator);

    return;   
};



bool FixWallHarmonic::prepareForRun() {
    // set this fix's evaulator with the appropriate parameters
    evaluator = EvaluatorWallHarmonic(k, dist);
    prepared = true;
    return prepared;
};


// export function
void export_FixWallHarmonic() {
	py::class_<FixWallHarmonic, SHARED(FixWallHarmonic), py::bases<FixWall>, boost::noncopyable > (
		"FixWallHarmonic",
		py::init<SHARED(State), std::string, std::string, Vector, Vector, real, real> (
			py::args("state", "handle", "groupHandle", "origin", "forceDir", "dist", "k")
		)
	)
	.def_readwrite("k", &FixWallHarmonic::k)
	.def_readwrite("dist", &FixWallHarmonic::dist)
	.def_readwrite("forceDir", &FixWallHarmonic::forceDir)
	.def_readwrite("origin", &FixWallHarmonic::origin)
	;
}


