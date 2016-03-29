#pragma once
#include "DataSet.h"
#include "GPUArray.h"
class Bounds;
void export_DataSetBounds();
class DataSetBounds : public DataSet {
    public:
		void collect(int64_t turn, BoundsGPU &, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *);
        void appendValues();
        DataSetBounds(uint32_t);
        std::vector<Bounds> vals;
        BoundsGPU stored;

};
