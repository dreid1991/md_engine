#include "DataComputer.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputer::DataComputer(State *state_, std::string computeMode_, bool requiresVirials_, std::string type_) : dataMultiple(1) {
    state = state_;
    computeMode = computeMode_;
    requiresVirials = requiresVirials_;
    lastGroupTag = 0;
    type = type_;
    requiresPerAtomVirials = false; //though may be set to true by a derived class
};


void DataComputer::prepareForRun() {
    if (computeMode=="scalar") {
        gpuBufferReduce = GPUArrayGlobal<real>(2);
        gpuBuffer = GPUArrayGlobal<real>(state->atoms.size() * dataMultiple);
    } else if (computeMode=="tensor") {
        gpuBufferReduce = GPUArrayGlobal<real>(2*6);
        gpuBuffer = GPUArrayGlobal<real>(state->atoms.size() * 6 * dataMultiple);
    } else if (computeMode=="vector") {
        gpuBuffer = GPUArrayGlobal<real>(state->atoms.size() * dataMultiple);
        gpuBufferReduce = GPUArrayGlobal<real>(); // initialize to empty
        sorted = std::vector<double>(state->atoms.size() * dataMultiple);
    } else {
        std::cout << "Invalid data type " << computeMode << ".  Must be scalar, tensor, or vector" << std::endl;
    }
}



void DataComputer::compute_GPU(bool transferToCPU, uint32_t groupTag) {
    if (computeMode=="scalar") {
        computeScalar_GPU(transferToCPU, groupTag);
    } else if (computeMode=="tensor") {
        computeTensor_GPU(transferToCPU, groupTag);
    } else if (computeMode=="vector") {
        computeVector_GPU(transferToCPU, groupTag);
    }
}



void DataComputer::compute_CPU() {
    if (computeMode=="scalar") {
        computeScalar_CPU();
    } else if (computeMode=="tensor") {
        computeTensor_CPU();
    } else if (computeMode=="vector") {
        computeVector_CPU();
    }
}



void DataComputer::appendData(py::list &vals) {
    if (computeMode=="scalar") {
        appendScalar(vals);
    } else if (computeMode=="tensor") {
        appendTensor(vals);
    } else if (computeMode=="vector") {
        appendVector(vals);
    }
}
