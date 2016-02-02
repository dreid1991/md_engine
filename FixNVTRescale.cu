#include "FixNVTRescale.h"
#include "cutils_func.h"


FixNVTRescale::FixNVTRescale(SHARED(State) state_, string handle_, string groupHandle_, boost::python::list intervals_, boost::python::list temps_, int applyEvery_) : Fix(state_, handle_, groupHandle_, NVTRescaleType, applyEvery_), curIdx(0), tempGPU(GPUArrayDevice<float>(1)), finished(false) {
    assert(boost::python::len(intervals_) == boost::python::len(temps_)); 
    assert(boost::python::len(intervals_) > 1);
    int len = boost::python::len(intervals_);
    for (int i=0; i<len; i++) {
        double interval = boost::python::extract<double>(intervals_[i]);
        double temp = boost::python::extract<double>(temps_[i]);
        intervals.push_back(interval);
        temps.push_back(temp);
    }

   assert(intervals[0] == 0 and intervals.back() == 1); 

}

bool FixNVTRescale::prepareForRun() {
    return true;
}

void __global__ rescale(int nAtoms, uint groupTag, float4 *vs, float4 *fs, float tempSet, float *tempCurPtr) {
    float tempCur = tempCurPtr[0] / nAtoms;
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


void FixNVTRescale::compute() {
    tempGPU.memset(0);
    int nAtoms = state->atoms.size();
    int turn = state->turn;
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
    sumVectorSqr3DTags<float, float4> <<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)>>>(tempGPU.ptr, gpd.vs(activeIdx), nAtoms, groupTag, gpd.fs(activeIdx));
    rescale<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, groupTag, gpd.vs(activeIdx), gpd.fs(activeIdx), temp, tempGPU.ptr);
}



bool FixNVTRescale::downloadFromRun() {
    finished = true;
    return true;
}


void export_FixNVTRescale() {
    class_<FixNVTRescale, SHARED(FixNVTRescale), bases<Fix> > ("FixNVTRescale", init<SHARED(State), string, string, boost::python::list, boost::python::list, optional<int> > (args("state", "handle", "groupHandle", "intervals", "temps", "applyEvery")))
        .def_readwrite("finished", &FixNVTRescale::finished)
        ;
}
