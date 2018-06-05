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

            // just do the processing of the cumulative histogram in postRun()
            void computeScalar_CPU(){};
            void computeVector_CPU(){}; 
            void computeTensor_CPU(){};

            DataComputerRDF(State *, std::string computeMode_, std::string species1, 
                            std::string species2, double binWidth);
            

            // prepareForRun() function - calls DataComputer::prepareForRun, and sets up auxiliary vectors
            // as required
            // --- computes number density of species1
            void prepareForRun();
            
            void postRun(boost::python::list &);

            GPUArrayGlobal<real> gpuBufferBins; //will be cast as virial if necessary

            std::vector<int> turns;

            double binWidth; // set in constructor
            double volume; //  set in prepareForRun()
            int nBins;     //  set in prepareForRun()
            double numberDensity; // number density of s2 in the sample volume
            int nTurns;    // number of turns we sampled our rdf from 
            int s1_count;
            int s2_count;

            GPUArrayDeviceGlobal<real> histogram;
            GPUArrayDeviceGlobal<real> cumulativeHistogram;
            std::string species1;
            std::string species2;
            int s1_type;
            int s2_type;
            // the vector that this returns will not be a list of values per-turn as usual;
            // instead, it will be 
            void appendScalar(boost::python::list &){};
            void appendVector(boost::python::list &){}; 
            void appendTensor(boost::python::list &){};


    };
};

