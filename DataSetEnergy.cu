#include "DataSetEnergy.h"
#include "cutils_func.h"
#include "boost_for_export.h"
using namespace std;
using namespace boost::python;

DataSetEnergy::DataSetEnergy(uint32_t groupTag_) : DataSet(groupTag_) {
    requiresEng = true;
}

void DataSetEnergy::collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *virials) {
    engGPU.d_data.memset(0);
    sumPlain<float, float> <<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*sizeof(float)>>>(engGPU.getDevData(), engs, nAtoms, groupTag, fs);
    engGPU.dataToHost();
    turns.push_back(turn);
    turnsPy.append(turn);
}
void DataSetEnergy::appendValues() {
    double engCur = (double) engGPU.h_data[0] / (double) engGPU.h_data[1]; 
    vals.push_back(engCur);
    valsPy.append(engCur);
    
}

void DataSetEnergy::prepareForRun() {
    engGPU = GPUArray<float>(2);
}

void export_DataSetEnergy() {
    class_<DataSetEnergy, SHARED(DataSetEnergy), bases<DataSet>, boost::noncopyable > ("DataSetEnergy", no_init)
        ;
}
