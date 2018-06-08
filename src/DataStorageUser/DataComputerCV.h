#pragma once

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
#include "Fix.h"

namespace MD_ENGINE {
    // data computer for constant volume heat capacity
    class DataComputerCV : public DataComputer {
        public:

            // we'll return the vector as f(t) from t(0)
            void computeScalar_GPU(bool, uint32_t){};
            void computeVector_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU(){};
            void computeVector_CPU();
            void computeTensor_CPU(){};

            DataComputerCV(State *, std::string computeMode_, std::string species);
            
            // prepareForRun() function - calls DataComputer::prepareForRun, and sets up auxiliary vectors
            // as required
            // --- computes number density of species
            void prepareForRun();
            void postRun(boost::python::list &);

            std::vector<int> turns;

            void appendScalar(boost::python::list &){};
            void appendVector(boost::python::list &){}; 
            void appendTensor(boost::python::list &){};


    };
};

