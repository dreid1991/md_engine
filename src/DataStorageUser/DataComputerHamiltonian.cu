#include "DataComputerHamiltonian.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;
const std::string computer_type_ = "hamiltonian";

DataComputerHamiltonian::DataComputerHamiltonian(State *state_, std::string computeMode_) : DataComputer(state_, computeMode_, false, computer_type_) {

    // computeMode can be either 'scalar' or 'vector', but not tensor.
    if ( (computeMode == "scalar") || (computeMode == "vector") ) {
        // do nothing
    } else {
       mdError("Fatal Error: DataComputerHamiltonian supports only the 'scalar' and 'vector' computeModes - please change to scalar or vector in your python script.");
    // don't need to check fixes; we compute all parts of the potential.
    }

    if (computeMode == "vector") {
        mdError("Fatal Error: DataComputerHamiltonian has not implemented the vector style yet!");
    }
}

void DataComputerHamiltonian::prepareForRun() {
    // call the DataComputer::prepareForRun() function
    DataComputer::prepareForRun();

    // dataMultiple is initialized in DataComputer constructor, always has a value of 1; integer type...
    // --- what is the point of this?
    if (computeMode=="scalar") {
        gpuBufferReduceKE = GPUArrayGlobal<real>(2);
        gpuBufferKE = GPUArrayGlobal<real>(state->atoms.size() * dataMultiple);
    } else if (computeMode=="vector") {
        gpuBufferKE = GPUArrayGlobal<real>(state->atoms.size() * dataMultiple);
    } else {
        // this will never be reached; computeMode is checked on instantiation to be either scalar or vector, 
        // for DataComputerHamiltonian.
        std::cout << "Invalid data type " << computeMode << ".  Must be scalar, tensor, or vector" << std::endl;
    }
}

void DataComputerHamiltonian::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
   
    // set the device data for potential energy and kinetic energies to zero
    gpuBuffer.d_data.memset(0);
    gpuBufferReduce.d_data.memset(0);

    gpuBufferKE.d_data.memset(0);
    gpuBufferReduceKE.d_data.memset(0);

    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;

    // iterate over the fixes and compute the potential 
    for (boost::shared_ptr<Fix> fix : fixes) {
        fix->setEvalWrapperMode("self");
        fix->setEvalWrapper();
        fix->singlePointEng(gpuBuffer.getDevData());
        fix->setEvalWrapperMode("offload");
        fix->setEvalWrapper();
    }

    // accumulate the per-particle energies in to a single total potential energy
     accumulate_gpu<real, real, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real)>>>
            (gpuBufferReduce.getDevData(), gpuBuffer.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingle());
   

     // accumulate the per-particle kinetic energies in to a single total kinetic energy
    accumulate_gpu<real, real4, 
        SumVectorSqr3DOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real)>>>
            (gpuBufferKE.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorSqr3DOverW());


    if (transferToCPU) {
        //does NOT sync
        gpuBufferKE.dataToHost();
        gpuBufferReduce.dataToHost();
    }

    /* NOTE
        At this point, we have grabbed:
            -the potential energy from all fixes for all atoms (gpuBufferReduce)
            -the kinetic energy of all atoms (gpuBufferKE)

        We still need to grab the following:
            -contributions from fixes (e.g., NVT or NPT variables if using MTK-type thermostats/barostats)
             or nothing (if NVE)
    */

}


void DataComputerHamiltonian::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    
    // as from DataComputerEnergy::computeVector_GPU()

    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();

    for (boost::shared_ptr<Fix> fix : fixes) {
        fix->setEvalWrapperMode("self");
        fix->setEvalWrapper();
        fix->singlePointEng(gpuBuffer.getDevData());
        fix->setEvalWrapperMode("offload");
        fix->setEvalWrapper();
    }
    if (transferToCPU) {
        gpuBuffer.dataToHost();
    }



}




void DataComputerHamiltonian::computeScalar_CPU() {
    
    // as from DataComputerTemperature; we just don't divide by the ndf
    double total = gpuBuffer.h_data[0];
    totalKEScalar = total * state->units.mvv_to_eng; 


    double totalPE = gpuBufferReduce.h_data[0];
    engScalar = totalPE;

    fromFixes = 0.0;

    /* TODO:
        for (boost::shared_ptr<Fix> fix : fixes) {
            double thisFixContribution = fix->getConservedQuantity();
            fromFixes += thisFixContribution;
       }
    */
    

}

void DataComputerHamiltonian::computeVector_CPU() {
    //ids have already been transferred, look in doDataComputation in integUtil
    std::vector<uint> &ids = state->gpd.ids.h_data;
    std::vector<real> &src = gpuBuffer.h_data;
    sortToCPUOrder(src, sorted, ids, state->gpd.idToIdxsOnCopy);
}



void DataComputerHamiltonian::appendScalar(boost::python::list &vals) {
    double total = 0.0;

    // PE
    total += engScalar;
    
    // KE
    total += totalKEScalar;

    // other
    total += fromFixes;

    vals.append(total);
}
void DataComputerHamiltonian::appendVector(boost::python::list &vals) {
    vals.append(sorted);
}

