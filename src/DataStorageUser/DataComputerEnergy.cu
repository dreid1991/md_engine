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
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    for (boost::shared_ptr<Fix> fix : fixes) {
        fix->setEvalWrapperOrig();
        fix->singlePointEng(gpuBuffer.d_data.data());
        fix->setEvalWrapper();
    }
    /*
    GPUArrayGlobal<float> &perParticleEng = gpd.perParticleEng;
    //printf("COPYING STUFF IN DATA COMPUTE ENG\n");
    //perParticleEng.dataToHost();
    //cudaDeviceSynchronize();
    //for (float x : perParticleEng.h_data) {
    //    printf("PARTICLE ENG %f\n", x);
   // }
    if (groupTag == 1) {
         accumulate_gpu<float, float, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBuffer.getDevData(), perParticleEng.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingle());
    } else {
        accumulate_gpu_if<float, float, SumSingleIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBuffer.getDevData(), perParticleEng.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingleIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        gpuBuffer.dataToHost();
    }
    */
}





void DataComputerEnergy::computeScalar_CPU() {
    int n;
    double total = gpuBuffer.h_data[0];
    if (lastGroupTag == 1) {
        n = state->atoms.size();//* (int *) &tempGPUScalar.h_data[1];
    } else {
        n = * (int *) &gpuBuffer.h_data[1];
    }
    engScalar = total / n;
}



void DataComputerEnergy::appendScalar(boost::python::list &vals) {
    vals.append(engScalar);
}

void DataComputerEnergy::prepareForRun() {
    if (fixes.size() == 0) {
        fixes = state->fixesShr; //if none specified, use them all
    }
    DataComputer::prepareForRun();
}

