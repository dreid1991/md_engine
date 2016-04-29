#pragma once
#ifndef DATASETTEMPERATURE_H
#define DATASETTEMPERATURE_H

#include "DataSet.h"
#include "GPUArray.h"

void export_DataSetTemperature();
class DataSetTemperature : public DataSet {
    public:
		void collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *, cudaDeviceProp &);
        void appendValues();
        DataSetTemperature(uint32_t);
        std::vector<double> vals;
        void prepareForRun();
        GPUArray<float> tempGPU;

};

#endif
