#include "FixLinearMomentum.h"
#include "State.h"
#include "cutils_func.h"

const std::string linearMomentumType = "LinearMomentum";

FixLinearMomentum::FixLinearMomentum(SHARED(State) state_, std::string handle_, std::string groupHandle_, int applyEvery_, Vector dimensions_)
  : Fix(state_, handle_, groupHandle_, linearMomentumType, false, false, false, applyEvery_), dimensions(dimensions_), sumMomentum(GPUArrayDeviceGlobal<real4>(2))
{   }

bool FixLinearMomentum::prepareForRun() {
    prepared = true;
    return prepared;
}


__global__ void rescale_all(int nAtoms, real4 *vs, real4 *sumData, real3 dims) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        real4 v = vs[idx];
        real4 sum = sumData[0];
        real invMassTotal = 1.0f / sum.w;
        v.x -= sum.x * invMassTotal * dims.x;
        v.y -= sum.y * invMassTotal * dims.y;
        v.z -= sum.z * invMassTotal * dims.z;
        vs[idx] = v;
    }
}


__global__ void rescale_group(int nAtoms, real4 *vs, real4 *fs, uint32_t groupTag, real4 *sumData, real3 dims) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint32_t tag = *(uint32_t *) &(fs[idx].w);
        if (tag & groupTag) {
            real4 v = vs[idx];
            real4 sum = sumData[0];
            real invMassTotal = 1.0f / sum.w;
            v.x -= sum.x * invMassTotal * dims.x;
            v.y -= sum.y * invMassTotal * dims.y;
            v.z -= sum.z * invMassTotal * dims.z;
            vs[idx] = v;
        }
    }
}

void FixLinearMomentum::compute(int virialMode) {
    real3 dimsreal3 = dimensions.asreal3();
    int nAtoms = state->atoms.size();
    real4 *vs = state->gpd.vs.getDevData();
    real4 *fs = state->gpd.vs.getDevData();
    int warpSize = state->devManager.prop.warpSize;

    sumMomentum.memset(0); 
    if (groupHandle == "all") {
        accumulate_gpu<real4, real4, SumVectorXYZOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real4)>>>
            (
             sumMomentum.data(),
             vs,
             nAtoms,
             warpSize,
             SumVectorXYZOverW()
            );
        rescale_all<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, vs, sumMomentum.data(), dimsreal3);


    } else {
        accumulate_gpu_if<real4, real4, SumVectorXYZOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real4)>>>
            (
             sumMomentum.data(),
             vs,
             nAtoms,
             warpSize,
             SumVectorXYZOverWIf(fs, groupTag)
            );
        rescale_group<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, vs, fs, groupTag, sumMomentum.data(), dimsreal3);
    }
}

void export_FixLinearMomentum() {
    boost::python::class_<FixLinearMomentum, SHARED(FixLinearMomentum),
        boost::python::bases<Fix> >("FixLinearMomentum",
            boost::python::init<SHARED(State), std::string, std::string, int, Vector> (
                boost::python::args("state", "handle", "groupHandle", "applyEvery", "dimensions")
            )
        )
    ;
}
