#include "FixNVTRescale.h"

#include "Bounds.h"
#include "cutils_func.h"
#include "State.h"

class SumVectorSqr3DOverWIf_Bounds {
public:
    float4 *fs;
    uint32_t groupTag;
    BoundsGPU bounds;
    SumVectorSqr3DOverWIf_Bounds(float4 *fs_, uint32_t groupTag_, BoundsGPU &bounds_) : fs(fs_), groupTag(groupTag_), bounds(bounds_) {}
    inline __host__ __device__ float process (float4 &velocity ) {
        return lengthSqrOverW(velocity);
    }
    inline __host__ __device__ float zero() {
        return 0;
    }
    inline __host__ __device__ bool willProcess(float4 *src, int idx) {
        float3 pos = make_float3(src[idx]);
        uint32_t atomGroupTag = * (uint32_t *) &(fs[idx].w);
        return (atomGroupTag & groupTag) && bounds.inBounds(pos);
    }
};

namespace py=boost::python;

using namespace std;

const std::string NVTRescaleType = "NVTRescale";
/*
__global__ void sumKeInBounds (float *dest, float4 *src, int n, unsigned int groupTag, float4 *fs, BoundsGPU bounds, int warpSize) {
    extern __shared__ float tmp[]; 
    int potentialIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if (potentialIdx < n) {
        unsigned int atomGroup = * (unsigned int *) &(fs[potentialIdx].w);
        if (atomGroup & groupTag) {
            float4 val = src[blockDim.x*blockIdx.x + threadIdx.x];
            if (bounds.inBounds(make_float3(val))) {
                tmp[threadIdx.x] = lengthSqrOverW( val ) ;
                atomicAdd(dest+1, 1);
            }
        } else {
            tmp[threadIdx.x] = 0;
        }
    } else {
        tmp[threadIdx.x] = 0;
    }
    __syncthreads();
    reduceByN(tmp, blockDim.x, warpSize);
    if (threadIdx.x == 0) {
        atomicAdd(dest, tmp[0]);
    }
}
*/

FixNVTRescale::FixNVTRescale(SHARED(State) state_, string handle_, string groupHandle_, py::list intervals_, py::list temps_, int applyEvery_, SHARED(Bounds) thermoBounds_)
    : Interpolator(intervals_, temps_), Fix(state_, handle_, groupHandle_, NVTRescaleType, false, false, false, applyEvery_),
      curIdx(0), tempGPU(GPUArrayDeviceGlobal<float>(2))
{
    thermoBounds = thermoBounds_;


}

FixNVTRescale::FixNVTRescale(SHARED(State) state_, string handle_, string groupHandle_, py::object tempFunc_, int applyEvery_, SHARED(Bounds) thermoBounds_)
    : Interpolator(tempFunc_), Fix(state_, handle_, groupHandle_, NVTRescaleType, false, false, false, applyEvery_),
      curIdx(0), tempGPU(GPUArrayDeviceGlobal<float>(2))
{
    thermoBounds = thermoBounds_;


}

FixNVTRescale::FixNVTRescale(SHARED(State) state_, string handle_, string groupHandle_, double constTemp_, int applyEvery_, SHARED(Bounds) thermoBounds_)
    : Interpolator(constTemp_), Fix(state_, handle_, groupHandle_, NVTRescaleType, false, false, false, applyEvery_),
      curIdx(0), tempGPU(GPUArrayDeviceGlobal<float>(2))
{
    thermoBounds = thermoBounds_;


}




bool FixNVTRescale::prepareForRun() {
    usingBounds = thermoBounds != SHARED(Bounds) (NULL);
    turnBeginRun = state->runInit;
    turnFinishRun = state->runInit + state->runningFor;
    if (usingBounds) {
        assert(state == thermoBounds->state);
        boundsGPU = thermoBounds->makeGPU();
    }
    return true;
}

void __global__ rescale(int nAtoms, uint groupTag, float4 *vs, float4 *fs, float tempSet, float *tempCurPtr) {
    int idx = GETIDX();
    float2 vals = ((float2 *) tempCurPtr)[0];
    float sumKe = vals.x;
    int n = * (int *) &(vals.y);
    if (vals.x > 0 and idx < nAtoms) {
        float tempCur = sumKe / n / 3.0f; //1th entry is #in group
        uint groupTagAtom = ((uint *) (fs+idx))[3];
        if (groupTag & groupTagAtom) {
            float4 vel = vs[idx];
            float w = vel.w;
            vel *= sqrtf(tempSet / tempCur);
            vel.w = w;
            vs[idx] = vel;
        }
    }
}


void __global__ rescaleInBounds(int nAtoms, uint groupTag, float4 *xs, float4 *vs, float4 *fs, float tempSet, float *tempCurPtr, BoundsGPU bounds) {
    int idx = GETIDX();
    float2 vals = ((float2 *) tempCurPtr)[0];
    float sumKe = vals.x;
    int n = * (int *) &(vals.y);
    if (vals.x > 0 and idx < nAtoms) {
        float tempCur = sumKe / n / 3.0f; //1th entry is #in group
        uint groupTagAtom = ((uint *) (fs+idx))[3];
        if (groupTag & groupTagAtom) {
            float3 x = make_float3(xs[idx]);
            if (bounds.inBounds(x)) {
                float4 vel = vs[idx];
                float w = vel.w;
                vel *= sqrtf(tempSet / tempCur);
                vel.w = w;
                vs[idx] = vel;
            }
        }
    }
}
/*
    template <class K, class T>
__global__ void SUMTESTS (K *dest, T *src, int n, unsigned int groupTag, float4 *fs, int warpSize) {
    extern __shared__ K tmp[];
    int potentialIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if (potentialIdx < n) {
        unsigned int atomGroup = * (unsigned int *) &(fs[potentialIdx].w);
        if (atomGroup & groupTag) {
            tmp[threadIdx.x] = lengthSqrOverW ( src[blockDim.x*blockIdx.x + threadIdx.x])  ;
            atomicAdd(dest+1, 1);
        } else {
            tmp[threadIdx.x] = 0;
        }
    } else {
        tmp[threadIdx.x] = 0;
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
        atomicAdd(dest, tmp[0]);
    }
}
*/


void FixNVTRescale::compute(bool computeVirials) {

    tempGPU.memset(0);
    int nAtoms = state->atoms.size();
    int64_t turn = state->turn;
    computeCurrentVal(turn);
    double temp = getCurrentVal();
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    int warpSize = state->devManager.prop.warpSize;
    if (usingBounds) {
        //should optimize this one in name #data per thread way
        accumulate_gpu_if<float, float4, SumVectorSqr3DOverWIf_Bounds, 4> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (
             tempGPU.data(), 
             gpd.vs(activeIdx), 
             nAtoms, 
             warpSize,
             SumVectorSqr3DOverWIf_Bounds(gpd.fs(activeIdx), groupTag, boundsGPU)
            );
        //sumKeInBounds<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)>>>(tempGPU.data(), gpd.vs(activeIdx), nAtoms, groupTag, gpd.fs(activeIdx), boundsGPU, warpSize);
        rescaleInBounds<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, groupTag, gpd.xs(activeIdx), gpd.vs(activeIdx), gpd.fs(activeIdx), temp, tempGPU.data(), boundsGPU);
    } else {
        //SUMTESTS<float, float4> <<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)>>>(tempGPU.data(), gpd.vs(activeIdx), nAtoms, groupTag, gpd.fs(activeIdx), warpSize);
        accumulate_gpu_if<float, float4, SumVectorSqr3DOverWIf, 4> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (
             tempGPU.data(), 
             gpd.vs(activeIdx), 
             nAtoms, 
             warpSize,
             SumVectorSqr3DOverWIf(gpd.fs(activeIdx), groupTag)
            );
        //sumVectorSqr3DTagsOverW<float, float4, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>(tempGPU.data(), gpd.vs(activeIdx), nAtoms, groupTag, gpd.fs(activeIdx), warpSize);
        //SAFECALL(sumVectorSqr3DTagsOverW<float, float4> <<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)>>>(tempGPU.data(), gpd.vs(activeIdx), nAtoms, groupTag, gpd.fs(activeIdx)));
        rescale<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, groupTag, gpd.vs(activeIdx), gpd.fs(activeIdx), temp, tempGPU.data());
        //SAFECALL(rescale<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, groupTag, gpd.vs(activeIdx), gpd.fs(activeIdx), temp, tempGPU.data()));
    }
}



bool FixNVTRescale::postRun() {
    finishRun();
    return true;
}




void export_FixNVTRescale() {
    py::class_<FixNVTRescale, SHARED(FixNVTRescale), py::bases<Fix> > (
        "FixNVTRescale", 
        py::init<boost::shared_ptr<State>, string, string, py::list, py::list, py::optional<int, boost::shared_ptr<Bounds> > >(
            py::args("state", "handle", "groupHandle", "intervals", "temps", "applyEvery", "thermoBounds")
            )

        
    )
   //HEY - ORDER IS IMPORTANT HERE.  LAST CONS ADDED IS CHECKED FIRST. A DOUBLE _CAN_ BE CAST AS A py::object, SO IF YOU PUT THE TEMPFUNC CONS LAST, CALLING WITH DOUBLE AS ARG WILL GO THERE, NOT TO CONST TEMP CONSTRUCTOR 
    .def(py::init<boost::shared_ptr<State>, string, string, py::object, py::optional<int, boost::shared_ptr<Bounds> > >(
                
            py::args("state", "handle", "groupHandle", "tempFunc", "applyEvery", "thermoBounds")
                )
            )
    .def(py::init<boost::shared_ptr<State>, string, string, double, py::optional<int, boost::shared_ptr<Bounds> > >(
            py::args("state", "handle", "groupHandle", "temp", "applyEvery", "thermoBounds")
                )
            )
    .def_readwrite("thermoBounds", &FixNVTRescale::thermoBounds);
    ;
}
