#include "DataComputerTemperature.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerTemperature::DataComputerTemperature(State *state_, bool computeScalar_, bool computeTensor_) : DataComputer(state_, computeScalar_, computeTensor_, false, false) {
}


void DataComputerTemperature::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    tempGPUScalar.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
         accumulate_gpu<float, float4, SumVectorSqr3DOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (tempGPUScalar.getDevData(), state->gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorSqr3DOverW());
    } else {
        accumulate_gpu_if<float, float4, SumVectorSqr3DOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (tempGPUScalar.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorSqr3DOverWIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        tempGPUScalar.dataToHost();
    }
}



void DataComputerTemperature::computeTensor_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    tempGPUTensor.d_data.memset(0); 
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
        accumulate_gpu<Virial, float4, SumVectorToVirialOverW, N_DATA_PER_THREAD>  <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            (tempGPUTensor.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorToVirialOverW());    
    } else {
        accumulate_gpu_if<Virial, float4, SumVectorToVirialOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            (tempGPUTensor.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorToVirialOverWIf(gpd.fs.getDevData(), groupTag));
    } 
    if (transferToCPU) {
        //does NOT sync
        tempGPUTensor.dataToHost();
    }
}

void DataComputerTemperature::computeScalar_CPU() {
    int n;
    double total = tempGPUScalar.h_data[0];
    if (lastGroupTag == 1) {
        n = state->atoms.size();//* (int *) &tempGPUScalar.h_data[1];
    } else {
        n = * (int *) &tempGPUScalar.h_data[1];
    }
    if (state->is2d) {
        ndf = 2*(n-1); //-1 is analagous to extra_dof in lammps
    } else {
        ndf = 3*(n-1);
    }
    totalKEScalar = total;
    tempScalar = state->units.mvv_to_eng * total / (state->units.boltz * ndf); 
}

void DataComputerTemperature::computeTensor_CPU() {
    Virial total = tempGPUTensor.h_data[0];
    total *= (state->units.mvv_to_eng / state->units.boltz);
    /*
    int n;
    if (lastGroupTag == 1) {
        n = state->atoms.size();
    } else {
        n = * (int *) &tempGPUTensor.h_data[1];
    }
    */
    tempTensor = total;
}

void DataComputerTemperature::computeTensorFromScalar() {
    int zeroDim = 3;
    if (state->is2d) {
        zeroDim = 2;
        tempTensor[0] = tempTensor[1] = totalKEScalar / 2.0;
    } else {
        tempTensor[0] = tempTensor[1] = tempTensor[2] = totalKEScalar / 3.0;
    }
    for (int i=zeroDim; i<6; i++) {
        tempTensor[i] = 0;
    }

}

void DataComputerTemperature::computeScalarFromTensor() {
    int n;
    if (lastGroupTag == 1) {
        n = state->atoms.size();//* (int *) &tempGPUScalar.h_data[1];
    } else {
        n = * (int *) &tempGPUScalar.h_data[1];
    }
    if (state->is2d) {
        ndf = 2*(n-1); //-1 is analagous to extra_dof in lammps
    } else {
        ndf = 3*(n-1);
    }
    totalKEScalar = tempTensor[0] + tempTensor[1] + tempTensor[2];
    tempScalar = totalKEScalar / ndf;


}

void DataComputerTemperature::appendScalar(boost::python::list &vals) {
    vals.append(tempScalar);
}
void DataComputerTemperature::appendTensor(boost::python::list &vals) {
    vals.append(tempTensor);
}

void DataComputerTemperature::prepareForRun() {
    if (computingScalar) {
        tempGPUScalar = GPUArrayGlobal<float>(2);
    }
    if (computingTensor) {
        tempGPUTensor = GPUArrayGlobal<Virial>(2);
    }
}

