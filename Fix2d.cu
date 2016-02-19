#include "Fix2d.h"
#include "State.h"

void __global__ compute_cu(float4 *xs, float4 *vs, float4 *fs, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        xs[idx].z = 0;
        vs[idx].z = 0;
        fs[idx].z = 0;

    }
}
//THIS NEEDS TO GO LAST


void Fix2d::compute() {
    //going to zero z in xs, vs, fs
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    compute_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(gpd.xs.getDevData(), gpd.vs.getDevData(), gpd.fs.getDevData(), nAtoms);
    
}

void export_Fix2d() {
    class_<Fix2d, SHARED(Fix2d), bases<Fix> > ("Fix2d", init<SHARED(State), string, int> (args("state", "handle", "applyEvery")))
        ;
}
