#pragma once
#ifndef DATACOMPUTERENERGY_H
#define DATACOMPUTERENERGY_H

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"

namespace MD_ENGINE {
    class DataComputerEnergy : public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU();
            void computeTensor_CPU(){};

            DataComputerEnergy(State *);
            void prepareForRun();
            double engScalar;
            //so these are just length 2 arrays.  First value is used for the result of the sum.  Second value is bit-cast to an int and used to cound how many values are present.
            GPUArrayGlobal<float> engGPUScalar;

            void appendScalar(boost::python::list &);
            void appendTensor(boost::python::list &){};


    };
};

#endif
