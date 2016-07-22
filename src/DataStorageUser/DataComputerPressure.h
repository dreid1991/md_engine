#pragma once
#ifndef DATACOMPUTERPRESSURE_H
#define DATACOMPUTERPRESSURE_H

#include "DataComputer.h"
#include "DataComputerTemperature.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"

namespace MD_ENGINE {
    class DataComputerPressure : public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t);

            void computeScalar_CPU();
            void computeTensor_CPU();

            DataComputerPressure(State *, bool, bool);
            void prepareForRun();
            double pressureScalar;
            Virial pressureTensor;
            //so these are just length 2 arrays.  First value is used for the result of the sum.  Second value is bit-cast to an int and used to cound how many values are present.
            GPUArrayGlobal<float> pressureGPUScalar;
            GPUArrayGlobal<Virial> pressureGPUTensor;

            void appendScalar(boost::python::list &);
            void appendTensor(boost::python::list &);

            double getScalar();
            Virial getTensor();
            DataComputerTemperature tempComputer;

    };
};

#endif
