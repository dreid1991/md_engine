#include "FixWallHarmonic.h"

#include "boost_for_export.h"
#include "cutils_math.h"
#include "State.h"

namespace py=boost::python;

const std::string wallHarmonicType = "WallHarmonic";

FixWallHarmonic::FixWallHarmonic(SHARED(State) state_, string handle_, string groupHandle_,
                                 Vector origin_, Vector forceDir_, double dist_, double k_)
    : Fix(state_, handle_, groupHandle_, wallHarmonicType, true, 1),
      origin(origin_), forceDir(forceDir_.normalized()), dist(dist_), k(k_)
{
    assert(dist >= 0);
}

void __global__ compute_cu(float4 *xs, int nAtoms, float4 *fs, float3 origin, float3 forceDir, float dist, float k, uint groupTag) {
    //forceDir is normalized in constructor
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 forceWhole = fs[idx];
        uint groupTagAtom = * (uint *) &forceWhole.w;
        if (groupTagAtom & groupTag) {
            float4 posWhole = xs[idx];
            float3 pos = make_float3(posWhole);
            float3 particleDist = pos - origin;
            float projection = dot(particleDist, forceDir);
            float magProj = cu_abs(projection);
            if (magProj <= dist) {
                float3 force = forceDir * ((dist - magProj) * k);
                float4 f = fs[idx];
                if (projection >= 0) {
                    f = f + force;
                } else {
                    f = f - force;
                }
                fs[idx] = f;
            }
        }

    }

}

void FixWallHarmonic::compute(bool computeVirials) {
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    int n = state->atoms.size();
    compute_cu<<<NBLOCK(n), PERBLOCK>>>(gpd.xs(activeIdx), n, gpd.fs(activeIdx), origin.asFloat3(), forceDir.asFloat3(), dist, k, groupTag);
}

void export_FixWallHarmonic() {
    py::class_<FixWallHarmonic, SHARED(FixWallHarmonic), py::bases<Fix> > (
        "FixWallHarmonic",
        py::init<SHARED(State), string, string, Vector, Vector, double, double> (
            py::args("state", "handle", "groupHandle", "origin", "forceDir", "dist", "k")
        )
    )
    .def_readwrite("k", &FixWallHarmonic::k)
    .def_readwrite("dist", &FixWallHarmonic::dist)
    .def_readwrite("forceDir", &FixWallHarmonic::forceDir)
    .def_readwrite("origin", &FixWallHarmonic::origin)
    ;

}
