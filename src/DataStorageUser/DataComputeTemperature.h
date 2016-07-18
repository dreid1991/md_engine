#pragma once
#ifndef DATASETTEMPERATURE_H
#define DATASETTEMPERATURE_H

#include "DataSet.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
class State;

void export_DataSetTemperature();
class DataSetTemperature : public DataSet {
    public:
		void collect();
        void appendValues();
        void computeScalar();
        void computeVector();
        DataSetTemperature(State *, uint32_t, bool, bool);
        std::vector<double> vals;
        void prepareForRun();
        GPUArrayGlobal<float> tempGPU;
        GPUArrayGlobal<Virial> tempGPUVec;

        double getScalar();
        std::vector<Virial> getVector();

};

#endif
