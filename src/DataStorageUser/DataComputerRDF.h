#pragma once

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
#include "Fix.h"

namespace MD_ENGINE {
    class DataComputerRDF : public DataComputer {
        public:

            // RDF only makes sense as a vector
            void computeScalar_GPU(bool, uint32_t){};
            void computeVector_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU(){};
            void computeVector_CPU();
            void computeTensor_CPU(){};

            // constructor; require A:B computation
            DataComputerRDF(State *, std::string computeMode_, std::string species1, std::string species2);
            
            // 

            // prepareForRun() function - calls DataComputer::prepareForRun, and sets up auxiliary vectors
            // as required
            void prepareForRun();
            
            GPUArrayGlobal<real> gpuBufferKE; //will be cast as virial if necessary
            GPUArrayGlobal<real> gpuBufferReduceKE; //target for reductions, also maybe cast as virial


            std::vector<int> turns;
            std::map<std::string, std::vector<double>> conserved_components;

            // the vector that this returns will not be a list of values per-turn as usual;
            // instead, it will be 
            void appendScalar(boost::python::list &){};
            void appendVector(boost::python::list &); 
            void appendTensor(boost::python::list &){};


    };
};

