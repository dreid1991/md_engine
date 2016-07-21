#include "DataSetTemperature.h"
#include "cutils_func.h"
#include "boost_for_export.h"
namespace py = boost::python;

DataSetTemperature::DataSetTemperature(State *state_, uint32_t groupTag_, bool computeScalar_, bool computeVirial_) : DataSet(state_, groupTag_, bool computeScalar_, bool computeVirial_) {
}

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


void DataSetTemperature::computeScalar(bool transferToCPU) {
    if (state->turn != turnScalarComputed) {
        GPUData &gpd = state->gpd;
        tempGPU.d_data.memset(0);
        accumulate_gpu_if<float, float4, SumVectorSqr3DOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
        (tempGPU.getDevData(), gpd.vs.getDevData(), nAtoms, state->deviceManager.prop.warpSize, SumVectorSqr3DOverWIf(gpd.fs.getDevData(), groupTag));
        turnScalarComputed = turn;
    } 
    if (transferToCPU and not lastScalarOnCPU) {
        //does NOT sync
        tempGPU.dataToHost();
        lastScalarOnCPU = true;
    }
}
void DataSetTemperature::computeVector(bool transferToCPU) {
    if (state->turn != turnVectorComputed) {
        GPUData &gpd = state->gpd;
        //tempGPUVec.d_data.memset(0); doing this in kernel
        ke_tensor <<<NBLOCK(nAtoms), PERBLOCK>>> (tempGPUVec.getDevData(), gpd.vs.getDevData(), nAtoms, gpd.fs.getDevData(), groupTag);
        turnVectorComputed = turn;
    } 
    if (transferToCPU and not lastVectorOnCPU) {
        //does NOT sync
        tempGPUVec.dataToHost();
        lastVectorOnCPU = true;
    }
}

double DataSetTemperature::getScalar() {
    int n = * (int *) &tempGPU.h_data[1];
    return tempGPU.h_data[0] / n / 3.0; 
}
vector<Virial> &DataSetTemperature::getVector() {
    return tempGPUVec.h_data;
}


//func to be called by integrator
void DataSetTemperature::collect() {
    if (computingScalar) {
        computeScalar(true);
    }
    if (computingVector) {
        computeVector(true);
    }
    turns.push_back(turn);
    turnsPy.append(turn);
}
void DataSetTemperature::appendValues() {
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

void DataSetTemperature::prepareForRun() {
    if (computingScalar) {
        tempGPU = GPUArrayGlobal<float>(2);
    }
    if (computingVector) {
        tempGPUVec = GPUArrayGlobal<Virial>(state->atoms.size());

    }
}

void export_DataSetTemperature() {
    py::class_<DataSetTemperature, SHARED(DataSetTemperature), bases<DataSet>, boost::noncopyable > ("DataSetTemperature", py::no_init)
        ;
}
