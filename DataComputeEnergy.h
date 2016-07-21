#pragma once
#ifndef DATASETENERGY_H
#define DATASETENERGY_H

#include "DataSet.h"
#include "GPUArrayGlobal.h"

void export_DataSetEnergy();
class DataSetEnergy : public DataSet {
    public:
		void collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *, cudaDeviceProp &);
        void appendValues();
        DataSetEnergy(uint32_t);
        std::vector<double> vals;
        void prepareForRun();
        GPUArrayGlobal<float> engGPU;

};

#endif
