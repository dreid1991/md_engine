#include "FixNVTRescale.h"
#include "cutils_func.h"

__global__ void sumKeInBounds (float *dest, float4 *src, int n, unsigned int groupTag, float4 *fs, BoundsGPU bounds) {
    extern __shared__ float tmp[]; /*should have length of # threads in a block (PERBLOCK) PLUS ONE for counting shared*/
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
    reduceByN(tmp, blockDim.x);
    if (threadIdx.x == 0) {
        atomicAdd(dest, tmp[0]);
    }
}


FixNVTRescale::FixNVTRescale(SHARED(State) state_, string handle_, string groupHandle_, boost::python::list intervals_, boost::python::list temps_, int applyEvery_, SHARED(Bounds) thermoBounds_ ) : Fix(state_, handle_, groupHandle_, NVTRescaleType, applyEvery_), curIdx(0), tempGPU(GPUArrayDevice<float>(2)), finished(false) {
    assert(boost::python::len(intervals_) == boost::python::len(temps_)); 
    assert(boost::python::len(intervals_) > 1);
    int len = boost::python::len(intervals_);
    for (int i=0; i<len; i++) {
        boost::python::extract<double> intPy(intervals_[i]);
        boost::python::extract<double> tempPy(temps_[i]);
        if (!intPy.check() or !tempPy.check()) {
            cout << "Invalid value given to fix with handle " << handle << endl;
            assert(intPy.check() and tempPy.check());
        }
        double interval = intPy;
        double temp = tempPy;
        intervals.push_back(interval);
        temps.push_back(temp);
    }
    thermoBounds = thermoBounds_;

   assert(intervals[0] == 0 and intervals.back() == 1); 

}

FixNVTRescale::FixNVTRescale(SHARED(State) state_, string handle_, string groupHandle_, vector<double> intervals_, vector<double> temps_, int applyEvery_, SHARED(Bounds) thermoBounds_) : Fix(state_, handle_, groupHandle_, NVTRescaleType, applyEvery_), curIdx(0), tempGPU(GPUArrayDevice<float>(2)), finished(false) {
    assert(intervals.size() == temps.size());
    intervals = intervals_;
    temps = temps_;
    thermoBounds = thermoBounds_;

    forceSingle = false;
}
bool FixNVTRescale::prepareForRun() {
    usingBounds = thermoBounds != SHARED(Bounds) (NULL);
    if (usingBounds) {
        assert(state == thermoBounds->state);
        boundsGPU = thermoBounds->makeGPU();
    }
    return true;
}

void __global__ rescale(int nAtoms, uint groupTag, float4 *vs, float4 *fs, float tempSet, float *tempCurPtr) {
    float2 vals = ((float2 *) tempCurPtr)[0];
    if (vals.x > 0) {
        float tempCur = vals.x / vals.y / 3.0f; //1th entry is #in group
        int idx = GETIDX();
        if (idx < nAtoms) {
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
}


void __global__ rescaleInBounds(int nAtoms, uint groupTag, float4 *xs, float4 *vs, float4 *fs, float tempSet, float *tempCurPtr, BoundsGPU bounds) {
    int idx = GETIDX();
    float2 vals = ((float2 *) tempCurPtr)[0];
    if (vals.x > 0 and idx < nAtoms) {
        float tempCur = vals.x / vals.y / 3.0f; //1th entry is #in group
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


void FixNVTRescale::compute(bool computeVirials) {

    tempGPU.memset(0);
    int nAtoms = state->atoms.size();
    int64_t turn = state->turn;
    double temp;
    if (finished) {
        temp = temps.back();
    } else {
        double frac = (turn-state->runInit) / (double) state->runningFor;
        while (frac > intervals[curIdx+1] and curIdx < intervals.size()-1) {
            curIdx++;
        }
        double tempA = temps[curIdx];
        double tempB = temps[curIdx+1];
        double intA = intervals[curIdx];
        double intB = intervals[curIdx+1];
        double fracThroughInterval = (frac-intA) / (intB-intA);
        temp = tempB*fracThroughInterval + tempA*(1-fracThroughInterval);
    }
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx;
    if (usingBounds) {
        sumKeInBounds<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)>>>(tempGPU.data(), gpd.vs(activeIdx), nAtoms, groupTag, gpd.fs(activeIdx), boundsGPU);
        rescaleInBounds<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, groupTag, gpd.xs(activeIdx), gpd.vs(activeIdx), gpd.fs(activeIdx), temp, tempGPU.data(), boundsGPU);
    } else {
        SAFECALL((sumVectorSqr3DTagsOverW<float, float4> <<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)>>>(tempGPU.data(), gpd.vs(activeIdx), nAtoms, groupTag, gpd.fs(activeIdx))), "sum");
        SAFECALL((rescale<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, groupTag, gpd.vs(activeIdx), gpd.fs(activeIdx), temp, tempGPU.data())), "rescale");
    }
}



bool FixNVTRescale::downloadFromRun() {
    finished = true;
    return true;
}


void export_FixNVTRescale() {
    boost::python::class_<FixNVTRescale,
                          SHARED(FixNVTRescale),
                          boost::python::bases<Fix> > (
        "FixNVTRescale",
        boost::python::init<SHARED(State), string, string, boost::python::list,
                            boost::python::list,
                            boost::python::optional<int, SHARED(Bounds)> > (
            boost::python::args("state", "handle", "groupHandle", "intervals",
                                "temps", "applyEvery", "thermoBounds")
        )
    )
    .def_readwrite("finished", &FixNVTRescale::finished)
    .def_readwrite("thermoBounds", &FixNVTRescale::thermoBounds);
    ;
}
