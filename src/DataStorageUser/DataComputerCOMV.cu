#include "DataComputerCOMV.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

// scalar, because we just need the one return - not a per-atom thing.
DataComputerCOMV::DataComputerCOMV(State *state_) : DataComputer(state_, "scalar", false) {

}


void DataComputerCOMV::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    sumMomentum.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();

    accumulate_gpu<float4, float4, SumVectorXYZOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float4)>>>
            (
             sumMomentum.getDevData(),
             gpd.vs.getDevData(),
             nAtoms,
             warpSize,
             SumVectorXYZOverW()
            );
    
    if (transferToCPU) {
        //does NOT sync
        sumMomentum.dataToHost();
    }
}


void DataComputerCOMV::prepareForRun() {
    DataComputer::prepareForRun();
    //then my own stuff
}


void DataComputerCOMV::computeScalar_CPU() {
    systemMomentum = * (float4 *) sumMomentum.h_data.data();


}

void DataComputerCOMV::appendScalar(boost::python::list &vals) {
    vals.append(systemMomentum);
}
