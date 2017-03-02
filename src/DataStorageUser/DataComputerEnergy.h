#pragma once

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
#include "Fix.h"

namespace MD_ENGINE {
    class DataComputerEnergy : public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t);
            void computeVector_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU();
            void computeVector_CPU();
            void computeTensor_CPU(){};

            DataComputerEnergy(State *, boost::python::list, std::string computeMode_);
            void prepareForRun();
            double engScalar;
            std::vector<double> engVector;
            //so these are just length 2 arrays.  First value is used for the result of the sum.  Second value is bit-cast to an int and used to cound how many values are present.

            std::vector<boost::shared_ptr<Fix> > fixes;

            void appendScalar(boost::python::list &);
            void appendVector(boost::python::list &); 
            void appendTensor(boost::python::list &){};


    };
};

