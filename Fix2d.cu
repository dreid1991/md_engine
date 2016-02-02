#include "Fix2d.h"
#include "State.h"

void __global__ compute_cu(cudaSurfaceObject_t xs, float4 *vs, float4 *fs, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        int xIdx = XIDX(idx, sizeof(float4));
        int yIdx = YIDX(idx, sizeof(float4));
        float4 x = surf2Dread<float4>(xs, xIdx*sizeof(float4), yIdx);
        x.z = 0;
        surf2Dwrite(x, xs, xIdx*sizeof(float4), yIdx);

        vs[idx].z = 0;
        fs[idx].z = 0;

    }
}
//THIS NEEDS TO GO LAST


void Fix2d::compute() {
    //going to zero z in xs, vs, fs
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx;
    compute_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.xs.getSurf(), gpd.vs(activeIdx), gpd.fs(activeIdx), nAtoms);
    
}

void export_Fix2d() {
    class_<Fix2d, SHARED(Fix2d), bases<Fix> > ("Fix2d", init<SHARED(State), string, int> (args("state", "handle", "applyEvery")))
        ;
}
