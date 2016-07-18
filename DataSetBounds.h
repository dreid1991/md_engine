#pragma once
#ifndef DATASETBOUNDS_H
#define DATASETBOUNDS_H

#include "DataSet.h"
#include "BoundsGPU.h"
class Bounds;
class State;
void export_DataSetBounds();
class DataSetBounds : public DataSet {
    public:
		void collect();
        void appendValues();
        void computeScalar();
        void computeVector() {};
        DataSetBounds(State *, uint32_t);
        std::vector<Bounds> vals;
        void prepareForRun();

        BoundsGPU stored;


};

#endif
