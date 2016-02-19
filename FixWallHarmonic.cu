#include "FixWallHarmonic.h"

FixWallHarmonic::FixWallHarmonic(SHARED(State) state_, string handle_, string groupHandle_, Vector origin_, Vector forceDir_, double dist_, double k_) : Fix(state_, handle_, groupHandle_, wallHarmonicType, 1), origin(origin_), forceDir(forceDir_.normalized()), dist(dist_), k(k_) {
    assert(dist >= 0);
    forceSingle = true;
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
            if (projection > 0 and projection <= dist) {
                float3 force = forceDir * ((dist - projection) * k);
                fs[idx] += force;
            }
        }

    }

}

void FixWallHarmonic::compute() {
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx;
    int n = state->atoms.size();
    compute_cu<<<NBLOCK(n), PERBLOCK>>>(gpd.xs(activeIdx), n, gpd.fs(activeIdx), origin.asFloat3(), forceDir.asFloat3(), dist, k, groupTag);
}

void export_FixWallHarmonic() {
    class_<FixWallHarmonic, SHARED(FixWallHarmonic), bases<Fix> > ("FixWallHarmonic", init<SHARED(State), string, string, Vector, Vector, double, double> (args("state", "handle", "groupHandle", "origin", "forceDir", "dist", "k")))
        .def_readwrite("k", &FixWallHarmonic::k)
        .def_readwrite("dist", &FixWallHarmonic::dist)
        .def_readwrite("forceDir", &FixWallHarmonic::forceDir)
        .def_readwrite("origin", &FixWallHarmonic::origin)
        ;

}
