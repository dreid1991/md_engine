#include "DataSetEnergy.h"
#include "cutils_func.h"
#include "boost_for_export.h"
using namespace std;
using namespace boost::python;

DataSetEnergy::DataSetEnergy(uint32_t groupTag_) : DataSet(groupTag_) {
    requiresEng = true;
}

void DataSetEnergy::collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *virials, cudaDeviceProp &prop) {
    engGPU.d_data.memset(0);
    accumulate_gpu_if<float, float, SumSingleIf, N_DATA_PER_THREAD><<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>(engGPU.getDevData(), engs, nAtoms, prop.warpSize, SumSingleIf(fs, groupTag));
   // sumPlain<float, float, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>(engGPU.getDevData(), engs, nAtoms, groupTag, fs, prop.warpSize);
    engGPU.dataToHost();
    turns.push_back(turn);
    turnsPy.append(turn);
}
void DataSetEnergy::appendValues() {
    double engCur = (double) engGPU.h_data[0];
    vals.push_back(engCur);
    valsPy.append(engCur);
    
}

void DataSetEnergy::prepareForRun() {
    engGPU = GPUArrayGlobal<float>(2);
}

void export_DataSetEnergy() {
    class_<DataSetEnergy, SHARED(DataSetEnergy), bases<DataSet>, boost::noncopyable > ("DataSetEnergy", no_init)
        ;
}
