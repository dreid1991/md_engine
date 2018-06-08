#pragma once

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
#include "Fix.h"

namespace MD_ENGINE {
    class DataComputerHamiltonian : public DataComputer {
        public:

            // computeScalar will assemble an array of contributions from individual sources, 
            // and sum them up to the total value of the conserved quantity
            void computeScalar_GPU(bool, uint32_t);

            // computeVector will assemble an array of contributions from individual sources 
            // and store them in a map of vectors
            void computeVector_GPU(bool, uint32_t);

            // computeTensor is not implemented
            void computeTensor_GPU(bool, uint32_t){};

            // assembles data on CPU side (computeScalar_CPU() and computeVector_CPU())
            void computeScalar_CPU();
            void computeVector_CPU();

            // not implemented
            void computeTensor_CPU(){};

            // constructor
            DataComputerHamiltonian(State *, std::string computeMode_);

            // prepareForRun() function - calls DataComputer::prepareForRun, and sets up auxiliary vectors
            // as required
            void prepareForRun();
 

            
            void postRun(boost::python::list &) {} ;
            // engScalar - instantaneous value of the potential energy
            double engScalar;
            
            // totalKEScalar; instantaneous value of the kinetic energy
            double totalKEScalar;

            // fromFixes; instantaneous value of the conserved quantity from fixes
            double fromFixes;

            // engVector - instantaneous value of the components of the conserved quantity
            std::map<std::string, double> engVector;
           
            // if in computeMode == "vector", then we need to keep track of the keys
            std::vector<std::string> map_keys;

            // See DataComputer.h - these are the arrays that will be used for potential energy
            // --- we will do sum of PE, KE on GPU;
            //     other terms (NVT, NPT) will be collected on CPU side each time from the pertinent fixes

            //GPUArrayGlobal<real> gpuBuffer; //will be cast as virial if necessary
            //GPUArrayGlobal<real> gpuBufferReduce; //target for reductions, also maybe cast as virial
            
            
            GPUArrayGlobal<real> gpuBufferKE; //will be cast as virial if necessary
            GPUArrayGlobal<real> gpuBufferReduceKE; //target for reductions, also maybe cast as virial

            std::vector<int> turns;
            // So, different geometric integrators have different conserved components;
            // e.g.:
            // NVE: Hamiltonian is KE + PE
            // NVT: Hamiltonian is (KE + PE) + (other terms)
            // NPT: Hamiltonian is (KE + PE) + (other terms)
            //
            // To see what other terms are, see, e.g.,
            // M. E. Tuckerman et. al., J Phys. A: Math. Gen. 39 (2006) 5629-5651
            //
            // The usual hamiltonian and a summation over the extended phase space variables
            //
            //
            //
            // ok


            // this is the vector that holds the 
            std::map<std::string, std::vector<double>> conserved_components;

            std::vector<boost::shared_ptr<Fix> > fixes;

            // appendScalar: the conserved quantity as a scalar
            void appendScalar(boost::python::list &);
            // appendVector: individual terms in a map
            void appendVector(boost::python::list &); 
            void appendTensor(boost::python::list &){};


    };
};

