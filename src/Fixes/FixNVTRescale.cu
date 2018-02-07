#include "FixNVTRescale.h"

#include "Bounds.h"
#include "cutils_func.h"
#include "State.h"

class SumVectorSqr3DOverWIf_Bounds {
public:
    real4 *fs;
    uint32_t groupTag;
    BoundsGPU bounds;
    SumVectorSqr3DOverWIf_Bounds(real4 *fs_, uint32_t groupTag_, BoundsGPU &bounds_) : fs(fs_), groupTag(groupTag_), bounds(bounds_) {}
    inline __host__ __device__ real process (real4 &velocity ) {
        return lengthSqrOverW(velocity);
    }
    inline __host__ __device__ real zero() {
        return 0;
    }
    inline __host__ __device__ bool willProcess(real4 *src, int idx) {
        real3 pos = make_real3(src[idx]);
        uint32_t atomGroupTag = * (uint32_t *) &(fs[idx].w);
        return (atomGroupTag & groupTag) && bounds.inBounds(pos);
    }
};

namespace py=boost::python;


const std::string NVTRescaleType = "NVTRescale";


FixNVTRescale::FixNVTRescale(SHARED(State) state_, std::string handle_, std::string groupHandle_, py::list intervals_, py::list temps_, int applyEvery_, int orderPreference_)
    : Interpolator(intervals_, temps_), Fix(state_, handle_, groupHandle_, NVTRescaleType, false, false, false, applyEvery_, orderPreference_),
      curIdx(0), tempComputer(state, "scalar")
{
    isThermostat = true;

}

FixNVTRescale::FixNVTRescale(SHARED(State) state_, std::string handle_, std::string groupHandle_, py::object tempFunc_, int applyEvery_, int orderPreference_)
    : Interpolator(tempFunc_), Fix(state_, handle_, groupHandle_, NVTRescaleType, false, false, false, applyEvery_, orderPreference_),
      curIdx(0), tempComputer(state, "scalar")
{
    isThermostat = true;


}

FixNVTRescale::FixNVTRescale(SHARED(State) state_, std::string handle_, std::string groupHandle_, double constTemp_, int applyEvery_, int orderPreference_)
    : Interpolator(constTemp_), Fix(state_, handle_, groupHandle_, NVTRescaleType, false, false, false, applyEvery_, orderPreference_),
      curIdx(0), tempComputer(state, "scalar")
{
    isThermostat = true;


}




bool FixNVTRescale::prepareFinal() {
    turnBeginRun = state->runInit;
    turnFinishRun = state->runInit + state->runningFor;
    tempComputer = MD_ENGINE::DataComputerTemperature(state,"scalar");
    tempComputer.prepareForRun();
    prepared = true;
    return prepared;
}

void __global__ rescale(int nAtoms, uint groupTag, real4 *vs, real4 *fs, real tempSet, real tempCur) {
    int idx = GETIDX();
    if (tempSet > 0 and idx < nAtoms) {
        uint groupTagAtom = ((uint *) (fs+idx))[3];
        if (groupTag & groupTagAtom) {
            real4 vel = vs[idx];
            real w = vel.w;
            vel *= sqrtf(tempSet / tempCur);
            vel.w = w;
            vs[idx] = vel;
        }
    }
}





//void FixNVTRescale::compute(int virialMode) {
bool FixNVTRescale::stepFinal() {
    tempComputer.computeScalar_GPU(true, groupTag);
    int nAtoms = state->atoms.size();
    int64_t turn = state->turn;
    computeCurrentVal(turn);
    double temp = getCurrentVal();
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();

    cudaDeviceSynchronize();
    tempComputer.computeScalar_CPU();
    rescale<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, groupTag, gpd.vs(activeIdx), gpd.fs(activeIdx), temp, tempComputer.tempScalar);

    return true;
}



bool FixNVTRescale::postRun() {
    finishRun();
    prepared = false;
    return true;
}


Interpolator *FixNVTRescale::getInterpolator(std::string type) {
    if (type == "temp") {
        return (Interpolator *) this;
    }
    return nullptr;
}


void export_FixNVTRescale() {
    py::class_<FixNVTRescale, SHARED(FixNVTRescale), py::bases<Fix>, boost::noncopyable > (
        "FixNVTRescale", 
        py::init<boost::shared_ptr<State>, std::string, std::string, py::list, py::list, py::optional<int > >(
            py::args("state", "handle", "groupHandle", "intervals", "temps", "applyEvery")
            )

        
    )
   //HEY - ORDER IS IMPORTANT HERE.  LAST CONS ADDED IS CHECKED FIRST. A DOUBLE _CAN_ BE CAST AS A py::object, SO IF YOU PUT THE TEMPFUNC CONS LAST, CALLING WITH DOUBLE AS ARG WILL GO THERE, NOT TO CONST TEMP CONSTRUCTOR 
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, py::object, py::optional<int > >(
                
            py::args("state", "handle", "groupHandle", "tempFunc", "applyEvery")
                )
            )
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, double, py::optional<int> >(
            py::args("state", "handle", "groupHandle", "temp", "applyEvery")
                )
            )
    ;
}
