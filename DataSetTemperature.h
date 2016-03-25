#pragma once
#include "DataSet.h"
#include "GPUArray.h"

void export_DataSetTemperature();
class DataSetTemperature : public DataSet {
    public:
		void collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *);
        DataSetTemperature(uint32_t);
        DataSetTemperature(){};
        std::vector<double> vals;
        GPUArray<float> tempGPU;

};
