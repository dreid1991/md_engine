#include "DataComputerTemperature.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerTemperature::DataComputerTemperature(State *state_, bool computeScalar_, bool computeVector_) : DataComputer(state_, computeScalar_, computeVector_) {
}


/*

__global__ void ke_tensor(Virial *virials, float4 *vs, int nAtoms, float4 *fs, uint32_t groupTag) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint32_t atomTag = *(uint32_t *) &(fs[idx].w);
        if (atomTag & groupTag) {
            float3 vel = make_float3(vs[idx]);
            Virial vir;
            vir.vals[0] = vel.x * vel.x;
            vir.vals[1] = vel.y * vel.y;
            vir.vals[2] = vel.z * vel.z;
            vir.vals[3] = vel.x * vel.y;
            vir.vals[4] = vel.x * vel.z;
            vir.vals[5] = vel.y * vel.z;
            virials[idx] = vir;

        } else {
            virials[idx] = Virial(0, 0, 0, 0, 0, 0);
        }
    }
}

*/
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
        accumulate_gpu<Virial, float4, SumVectorToVirial, N_DATA_PER_THREAD>  <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            (tempGPUTensor.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorToVirial());    
    } else {
        accumulate_gpu_if<Virial, float4, SumVectorToVirialIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            (tempGPUTensor.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorToVirialIf(gpd.fs.getDevData(), groupTag));
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
    tempScalar = total / (3*n); //deal with dimensionality, also fix DOF
}

void DataComputerTemperature::computeTensor_CPU() {
    int n;
    Virial total = tempGPUTensor.h_data[0];
    if (lastGroupTag == 1) {
        n = state->atoms.size();//* (int *) &tempGPUScalar.h_data[1];
    } else {
        n = * (int *) &tempGPUTensor.h_data[1];
    }
    total /= n;
    tempTensor = total;
}

void DataComputerTemperature::appendScalar(boost::python::list &vals) {
    vals.append(tempScalar);
}
void DataComputerTemperature::appendTensor(boost::python::list &vals) {
    vals.append(tempTensor);
}
/*
void DataComputerTemperature::collect() {
    if (computingScalar) {
        computeScalar(true);
    }
    if (computingVector) {
        computeVector(true);
    }
    turns.push_back(turn);
    turnsPy.append(turn);
}
void DataComputerTemperature::appendValues() {
    if (computeScalar) {
        double tempCur = getScalar();
        vals.push_back(tempCur);
        valsPy.append(tempCur);
    } 
    if (computeVector) {
        //store in std::vector too?
        std::vector<Virial> &virials = getVector();
        vectorsPy.append(virials);

    }
    //reset lastScalar... bools
    
}
*/
void DataComputerTemperature::prepareForRun() {
    if (computingScalar) {
        tempGPUScalar = GPUArrayGlobal<float>(2);
    }
    if (computingTensor) {
        tempGPUTensor = GPUArrayGlobal<Virial>(2);
    }
}

