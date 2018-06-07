#pragma once

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
#include "Fix.h"

namespace MD_ENGINE {
    class DataComputerDiffusion : public DataComputer {
        public:

            // we'll return the vector as f(t) from t(0)
            void computeScalar_GPU(bool, uint32_t){};
            void computeVector_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU(){};
            void computeVector_CPU();
            void computeTensor_CPU(){};

            DataComputerDiffusion(State *, std::string computeMode_, std::string species);
            

            // prepareForRun() function - calls DataComputer::prepareForRun, and sets up auxiliary vectors
            // as required
            // --- computes number density of species
            void prepareForRun();
            
            void postRun(boost::python::list &);

            GPUArrayDeviceGlobal<real4> xs_init;        // positions at t = 0 for calculating rmsd
            GPUArrayDeviceGlobal<real4> xs_recent;      // save previous coordinates, decide if we need to unwrap
            GPUArrayDeviceGlobal<real4> boxes_traveled; // for unwrapping periodicities; initialize to (0,0,0,0);
            GPUArrayDeviceGlobal<real> diffusion_scalar;
        
            // note that this is returned in A^2/ps (typical units for water)
            std::vector<double> diffusion_vector;

            std::string species;
            int species_type;
            int species_count;
            std::vector<int> turns;

            void appendScalar(boost::python::list &){};
            void appendVector(boost::python::list &){}; 
            void appendTensor(boost::python::list &){};


    };
};

