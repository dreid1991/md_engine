#include "DataComputerEnergy.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerEnergy::DataComputerEnergy(State *state_, py::list fixes_, std::string computeMode_) : DataComputer(state_, computeMode_, false) {
    if (py::len(fixes_)) {
        int len = py::len(fixes_);
        for (int i=0; i<len; i++) {
            py::extract<boost::shared_ptr<Fix> > fixPy(fixes_[i]);
            if (!fixPy.check()) {
                assert(fixPy.check());
            }
            fixes.push_back(fixPy);
        }
    }
}


void DataComputerEnergy::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    gpuBuffer.d_data.memset(0);
    gpuBufferReduce.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    for (boost::shared_ptr<Fix> fix : fixes) {
        std::cout << "handle " << fix->handle << std::endl;
        fix->setEvalWrapperOrig();
        fix->singlePointEng(gpuBuffer.getDevData());
        fix->setEvalWrapper();
    }
    if (groupTag == 1) {
         accumulate_gpu<float, float, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBufferReduce.getDevData(), gpuBuffer.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingle());
    } else {
        accumulate_gpu_if<float, float, SumSingleIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBufferReduce.getDevData(), gpuBuffer.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingleIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        gpuBufferReduce.dataToHost();
    }
}


void DataComputerEnergy::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    for (boost::shared_ptr<Fix> fix : fixes) {
        fix->setEvalWrapperOrig();
        fix->singlePointEng(gpuBuffer.getDevData());
        fix->setEvalWrapper();
    }
    if (transferToCPU) {
        //does NOT sync
        gpuBuffer.dataToHost();
        gpd.ids.dataToHost(); //need ids to map back to original ordering
    }
}




void DataComputerEnergy::computeScalar_CPU() {
    int n;
    double total = gpuBufferReduce.h_data[0];
    if (lastGroupTag == 1) {
        n = state->atoms.size();//* (int *) &tempGPUScalar.h_data[1];
    } else {
        n = * (int *) &gpuBufferReduce.h_data[1];
    }
    printf("total %f n %d\n", total, n);
    engScalar = total / n;
}

void DataComputerEnergy::computeVector_CPU() {
    std::vector<uint> &ids = state->gpd.ids.h_data;
    std::vector<float> &src = gpuBuffer.h_data;
    sortToCPUOrder(src, sorted, ids, state->gpd.idToIdxsOnCopy);
}



void DataComputerEnergy::appendScalar(boost::python::list &vals) {
    vals.append(engScalar);
}
void DataComputerEnergy::appendVector(boost::python::list &vals) {
    vals.append(sorted);
}

void DataComputerEnergy::prepareForRun() {
    if (fixes.size() == 0) {
        fixes = state->fixesShr; //if none specified, use them all
    }
    DataComputer::prepareForRun();
}

