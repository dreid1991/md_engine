#include "DataSetTemperature.h"
#include "cutils_func.h"


DataSetTemperature::DataSetTemperature(uint32_t groupTag_) : DataSet(groupTag_) {
    tempGPU = GPUArray<float>(2);
}

void collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *virials) {
    tempGPU.d_data.memset(0);
    sumVectorSqr3DTagsOverW<float, float4> <<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)+1>>>(tempGPU.getDevData(), vs, nAtoms, groupTag, fs);
    tempGPU.dataToDevice();
    double tempCur = tempGPU.h_data[0] / tempGPU.h_data[1] / 3.0; 
    vals.push_back(tempCur);
    turns.push_back(turn);
}

