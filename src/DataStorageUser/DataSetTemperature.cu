#include "DataSetTemperature.h"
#include "cutils_func.h"
#include "boost_for_export.h"
using namespace std;
using namespace boost::python;

DataSetTemperature::DataSetTemperature(uint32_t groupTag_) : DataSet(groupTag_) {
}




void DataSetTemperature::collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *virials, cudaDeviceProp &prop) {
    tempGPU.d_data.memset(0);
    accumulate_gpu_if<float, float4, SumVectorSqr3DOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
        (tempGPU.getDevData(), vs, nAtoms, prop.warpSize, SumVectorSqr3DOverWIf(fs, groupTag));
    //sumVectorSqr3DTagsOverW<float, float4, 1> <<<NBLOCK(nAtoms / (double) 1), PERBLOCK, 1*PERBLOCK*sizeof(float)>>>(tempGPU.getDevData(), vs, nAtoms, groupTag, fs, prop.warpSize);
    tempGPU.dataToHost();
    turns.push_back(turn);
    turnsPy.append(turn);
}
void DataSetTemperature::appendValues() {
    int n = * (int *) &tempGPU.h_data[1];
    double tempCur = (double) tempGPU.h_data[0] / n / 3.0; 
    vals.push_back(tempCur);
    valsPy.append(tempCur);
    
}

void DataSetTemperature::prepareForRun() {
    tempGPU = GPUArrayGlobal<float>(2);
}

void export_DataSetTemperature() {
    class_<DataSetTemperature, SHARED(DataSetTemperature), bases<DataSet>, boost::noncopyable > ("DataSetTemperature", no_init)
        ;
}
