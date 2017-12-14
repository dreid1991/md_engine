#include "DataComputerHamiltonian.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerHamiltonian::DataComputerHamiltonian(State *state_, std::string computeMode_) : DataComputer(state_, computeMode_, false) {

    // don't need to check fixes; we compute all parts of the potential.
}

void DataComputerHamiltonian::prepareForRun() {
    // call the DataComputer::prepareForRun() function
    DataComputer::prepareForRun();

    // dataMultiple is initialized in DataComputer constructor, always has a value of 1; integer type...
    // --- what is the point of this?
    if (computeMode=="scalar") {
        gpuBufferReduceKE = GPUArrayGlobal<real>(2);
        gpuBufferKE = GPUArrayGlobal<real>(state->atoms.size() * dataMultiple);
    } else if (computeMode=="tensor") {
        gpuBufferReduceKE = GPUArrayGlobal<real>(2*6);
        gpuBufferKE = GPUArrayGlobal<real>(state->atoms.size() * 6 * dataMultiple);
    } else if (computeMode=="vector") {
        gpuBufferKE = GPUArrayGlobal<real>(state->atoms.size() * dataMultiple);
        sortedKE = std::vector<double>(state->atoms.size() * dataMultiple);
    } else {
        std::cout << "Invalid data type " << computeMode << ".  Must be scalar, tensor, or vector" << std::endl;
    }

}

void DataComputerHamiltonian::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    
    gpuBuffer.d_data.memset(0);
    gpuBufferReduce.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    for (boost::shared_ptr<Fix> fix : fixes) {
        fix->setEvalWrapperMode("self");
        fix->setEvalWrapper();
        if (otherIsAll) {
            fix->singlePointEng(gpuBuffer.getDevData());
        } else {
            fix->singlePointEngGroupGroup(gpuBuffer.getDevData(), groupTag, groupTagB);

        }
        fix->setEvalWrapperMode("offload");
        fix->setEvalWrapper();
    }
    if (groupTag == 1 or !otherIsAll) { //if other isn't all, then only group-group energies got computed so need to sum them all up anyway.  If other is all then every eng gets computed so need to accumulate only things in group
         accumulate_gpu<real, real, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real)>>>
            (gpuBufferReduce.getDevData(), gpuBuffer.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingle());
    } else {
        accumulate_gpu_if<real, real, SumSingleIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real)>>>
            (gpuBufferReduce.getDevData(), gpuBuffer.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingleIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        gpuBufferReduce.dataToHost();
    }
}


void DataComputerHamiltonian::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();

    for (boost::shared_ptr<Fix> fix : fixes) {
        fix->setEvalWrapperMode("self");
        fix->setEvalWrapper();
        if (otherIsAll) {
            fix->singlePointEng(gpuBuffer.getDevData());
        } else {
            fix->singlePointEngGroupGroup(gpuBuffer.getDevData(), groupTag, groupTagB);
        }
        fix->setEvalWrapperMode("offload");
        fix->setEvalWrapper();
    }
    if (transferToCPU) {
        gpuBuffer.dataToHost();
    }
}




void DataComputerHamiltonian::computeScalar_CPU() {
    //int n;
    double total = gpuBufferReduce.h_data[0];
    /*
    if (lastGroupTag == 1) {
        n = state->atoms.size();//* (int *) &tempGPUScalar.h_data[1];
    } else {
        n = * (int *) &gpuBufferReduce.h_data[1];
    }
    */
    //just going with total energy value, not average
    engScalar = total;
}

void DataComputerHamiltonian::computeVector_CPU() {
    //ids have already been transferred, look in doDataComputation in integUtil
    std::vector<uint> &ids = state->gpd.ids.h_data;
    std::vector<real> &src = gpuBuffer.h_data;
    sortToCPUOrder(src, sorted, ids, state->gpd.idToIdxsOnCopy);
}



void DataComputerHamiltonian::appendScalar(boost::python::list &vals) {
    vals.append(engScalar);
}
void DataComputerHamiltonian::appendVector(boost::python::list &vals) {
    vals.append(sorted);
}

void DataComputerHamiltonian::prepareForRun() {
    if (fixes.size() == 0) {
        fixes = state->fixesShr; //if none specified, use them all
    } else {
        //make sure that fixes are activated
        for (auto fix : fixes) {
            bool found = false;
            for (auto fix2 : state->fixesShr) {
                if (fix->handle == fix2->handle && fix->type == fix2->type) {
                    found = true;
                }
            }
            if (not found) {
                std::cout << "Trying to record energy for inactive fix " << fix->handle << ".  Quitting" << std::endl;
                assert(found);
            }
        }
    }
    DataComputer::prepareForRun();
}

