#include "DataComputerEnergy.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerEnergy::DataComputerEnergy(State *state_) : DataComputer(state_, true, false, true, false) {
}


void DataComputerEnergy::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    engGPUScalar.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    GPUArrayGlobal<float> &perParticleEng = gpd.perParticleEng;
    if (groupTag == 1) {
         accumulate_gpu<float, float, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (engGPUScalar.getDevData(), perParticleEng.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingle());
    } else {
        accumulate_gpu_if<float, float, SumSingleIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (engGPUScalar.getDevData(), perParticleEng.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingleIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        engGPUScalar.dataToHost();
    }
}





void DataComputerEnergy::computeScalar_CPU() {
    int n;
    double total = engGPUScalar.h_data[0];
    if (lastGroupTag == 1) {
        n = state->atoms.size();//* (int *) &tempGPUScalar.h_data[1];
    } else {
        n = * (int *) &engGPUScalar.h_data[1];
    }
    engScalar = total / n;
}



void DataComputerEnergy::appendScalar(boost::python::list &vals) {
    vals.append(engScalar);
}

void DataComputerEnergy::prepareForRun() {
        engGPUScalar = GPUArrayGlobal<float>(2);
}

