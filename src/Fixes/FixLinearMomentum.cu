#include "FixLinearMomentum.h"
#include "State.h"
#include "cutils_func.h"

const std::string linearMomentumType = "LinearMomentum";

FixLinearMomentum::FixLinearMomentum(SHARED(State) state_, string handle_, string groupHandle_, int applyEvery_, Vector dimensions_): Fix(state_, handle_, groupHandle_, linearMomentumType, applyEvery_), dimensions(dimensions_), sumMomentum(GPUArrayDeviceGlobal<float4>(2)) {
}

    template <class K, class T>
__global__ void NAME (K *dest, T *src, int n, unsigned int groupTag, float4 *fs, int warpSize) {
    extern __shared__ K tmp[]; /*should have length of # threads in a block (PERBLOCK)  */
    int potentialIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if (potentialIdx < n) {
        unsigned int atomGroup = * (unsigned int *) &(fs[potentialIdx].w);
        if (atomGroup & groupTag) {
            tmp[threadIdx.x] = ( xyzOverW(src[blockDim.x*blockIdx.x + threadIdx.x]) ) ;
            atomicAdd((int *) (dest+1), 1);/*I TRIED DOING ATOMIC ADD IN SHARED MEMORY, BUT IT SET A BUNCH OF THE OTHER SHARED MEMORY VALUES TO ZERO.  VERY CONFUSING*/
        } else {
            tmp[threadIdx.x] = K();
        }
    } else {
        tmp[threadIdx.x] = K();
    }
    __syncthreads();
    int curLookahead = 1;
    int maxLookahead = log2f(blockDim.x-1);
    for (int i=0; i<=maxLookahead; i++) {
        if (! (threadIdx.x % (curLookahead*2))) {
            tmp[threadIdx.x] += tmp[threadIdx.x + curLookahead];
        }
        curLookahead *= 2;
        if (curLookahead >= warpSize) {
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(((float *) dest), tmp[0].x);
        atomicAdd(((float *) dest) + 1, tmp[0].y);
        atomicAdd(((float *) dest) + 2, tmp[0].z);
    }
}
bool FixLinearMomentum::prepareForRun() {
    return true;
}
void FixLinearMomentum::compute(bool computeVirials) {
    float3 dimsFloat3 = dimensions.asFloat3();
    int nAtoms = state->atoms.size();
    float4 *vs = state->gpd.vs.getDevData();
    float4 *fs = state->gpd.vs.getDevData();
    int warpSize = state->devManager.prop.warpSize;
    NAME<float4, float4> <<<NBLOCK(nAtoms), PERBLOCK, sizeof(float4) * PERBLOCK>>>(sumMomentum.data(), vs, groupTag, nAtoms, fs, warpSize);
}
void export_FixLinearMomentum() {
    boost::python::class_<FixLinearMomentum, SHARED(FixLinearMomentum), boost::python::bases<Fix> >  ("FixLinearMomentum", boost::python::init<SHARED(State), string, string, int, Vector> (
                boost::python::args("state", "handle", "groupHandle", "applyEvery", "dimensions")
                )
            )
        ;
}
