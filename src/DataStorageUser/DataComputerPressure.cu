#include "DataComputerPressure.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerPressure::DataComputerPressure(State *state_, bool computeScalar_, bool computeTensor_) : DataComputer(state_, computeScalar_, computeTensor_, false, true), tempComputer(state_, computeScalar_, computeTensor_) {
}


void DataComputerPressure::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    mdAssert(groupTag == 1, "Trying to compute pressure for group other than 'all'");
    tempComputer.computeScalar_GPU(transferToCPU, groupTag);
    GPUData &gpd = state->gpd;
    pressureGPUScalar.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
         accumulate_gpu<float, Virial, SumVirialToScalar, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (pressureGPUScalar.getDevData(), gpd.virials.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVirialToScalar());
    } else {
        accumulate_gpu_if<float, Virial, SumVirialToScalarIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (pressureGPUScalar.getDevData(), 
             gpd.virials.getDevData(), 
             nAtoms, 
             state->devManager.prop.warpSize, 
             SumVirialToScalarIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        pressureGPUScalar.dataToHost();
    }
}



void DataComputerPressure::computeTensor_GPU(bool transferToCPU, uint32_t groupTag) {
    mdAssert(groupTag == 1, "Trying to compute pressure for group other than 'all'");
    tempComputer.computeTensor_GPU(transferToCPU, groupTag);
    GPUData &gpd = state->gpd;
    pressureGPUTensor.d_data.memset(0); 
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
        accumulate_gpu<Virial, Virial, SumVirial, N_DATA_PER_THREAD>  <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            (pressureGPUTensor.getDevData(), gpd.virials.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVirial());    
    } else {
        accumulate_gpu_if<Virial, Virial, SumVirialIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            (pressureGPUTensor.getDevData(), gpd.virials.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVirialIf(gpd.fs.getDevData(), groupTag));
    } 
    if (transferToCPU) {
        //does NOT sync
        pressureGPUTensor.dataToHost();
    }
}

void DataComputerPressure::computeScalar_CPU() {
    //we are assuming that z component of virial is zero if sim is 2D
    tempComputer.computeScalar_CPU();
    double sumVirial = pressureGPUScalar.h_data[0];
    double dim = state->is2d ? 2 : 3;
    double volume = state->boundsGPU.volume();
    pressureScalar = (tempComputer.tempScalar * tempComputer.ndf + sumVirial) / (dim * volume);
}

void DataComputerPressure::computeTensor_CPU() {
    tempComputer.computeTensor_CPU();
    Virial tempTensor = tempComputer.tempTensor;
    pressureTensor = Virial(0, 0, 0, 0, 0, 0);
    Virial sumVirial = pressureGPUTensor.h_data[0];
    double volume = state->boundsGPU.volume();
    for (int i=0; i<6; i++) {
        pressureTensor[i] = (tempTensor[i] + sumVirial[i]) / volume;
    }
    if (state->is2d) {
        pressureTensor[2] = 0;
        pressureTensor[4] = 0;
        pressureTensor[5] = 0;
    }
}

void DataComputerPressure::appendScalar(boost::python::list &vals) {
    vals.append(pressureScalar);
}
void DataComputerPressure::appendTensor(boost::python::list &vals) {
    vals.append(pressureTensor);
}

void DataComputerPressure::prepareForRun() {
    tempComputer = DataComputerTemperature(state, computingScalar, computingTensor);
    tempComputer.prepareForRun();
    if (computingScalar) {
        pressureGPUScalar = GPUArrayGlobal<float>(2);
    }
    if (computingTensor) {
        pressureGPUTensor = GPUArrayGlobal<Virial>(2);
    }
}

