#include "DataComputerTemperature.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
#include "Fix.h"
#include "Group.h"

namespace py = boost::python;
using namespace MD_ENGINE;
const std::string computer_type_ = "temperature";



namespace { 

__global__ void computePerParticleMVV(int nAtoms, 
                                  const real4 *__restrict__  vs,
                                  real * __restrict__ mvv)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        real4 vel_whole = vs[idx];
        real3 vel = make_real3(vel_whole);
        real invmass = vel_whole.w;
        if (invmass < INVMASSBOOL) {
            mvv[idx] = dot(vel,vel) / invmass;
        } else {
            mvv[idx] = 0.0;
        }
    }
}
} // namespace

DataComputerTemperature::DataComputerTemperature(State *state_, std::string computeMode_) : DataComputer(state_, computeMode_, false,computer_type_) {

}


void DataComputerTemperature::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
         accumulate_gpu<real, real4, SumVectorSqr3DOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real)>>>
            (gpuBuffer.getDevData(), state->gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorSqr3DOverW());
    } else {
        accumulate_gpu_if<real, real4, SumVectorSqr3DOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real)>>>
            (gpuBuffer.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorSqr3DOverWIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        gpuBuffer.dataToHost();
    }
}


void DataComputerTemperature::prepareForRun() {
    DataComputer::prepareForRun();
}


void DataComputerTemperature::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();

    /*
    oneToOne_gpu<real, real4, SumVectorSqr3DOverW, 8> <<< NBLOCK(nAtoms / (double) 8), PERBLOCK>>> 
            (gpuBuffer.getDevData(), gpd.vs.getDevData(), nAtoms, SumVectorSqr3DOverW());

    */

    computePerParticleMVV<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms,
                                                        gpd.vs.getDevData(),
                                                        gpuBuffer.getDevData());

    /*
    if (state->units.unitType == UNITS::REAL) {

        convertKE<<<NBLOCK(nAtoms),PERBLOCK>>>(nAtoms,
                                               gpuBuffer.getDevData(),


    }
    */
    if (transferToCPU) {
        gpuBuffer.dataToHost();
        gpd.ids.dataToHost();
    }

}

void DataComputerTemperature::computeTensor_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    gpuBuffer.d_data.memset(0); 
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
        accumulate_gpu<Virial, real4, SumVectorToVirialOverW, N_DATA_PER_THREAD>  <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            ((Virial *) gpuBuffer.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorToVirialOverW());    
    } else {
        accumulate_gpu_if<Virial, real4, SumVectorToVirialOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            ((Virial *) gpuBuffer.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorToVirialOverWIf(gpd.fs.getDevData(), groupTag));
    } 
    if (transferToCPU) {
        //does NOT sync
        gpuBuffer.dataToHost();
    }
}

void DataComputerTemperature::computeScalar_CPU() {
    
    //int n;
    double total = gpuBuffer.h_data[0];
    Group &thisGroup = state->groups[lastGroupTag];

    ndf = thisGroup.getNDF();
    
    totalKEScalar = total * state->units.mvv_to_eng; 
    tempScalar = state->units.mvv_to_eng * total / (state->units.boltz * ndf); 
}


void DataComputerTemperature::computeVector_CPU() {
    //appending members in group in no meaningful order
    //std::vector<real> &kes = gpuBuffer.h_data;
    //std::vector<uint> &ids = state->gpd.ids.h_data;
    //std::vector<int> &idToIdxOnCopy = state->gpd.idToIdxsOnCopy;
    std::vector<Atom> &atoms = state->atoms;
    //sortToCPUOrder(kes, sorted, ids, state->gpd.idToIdxsOnCopy);
    
    std::vector<uint> &ids = state->gpd.ids.h_data;
    std::vector<real> &src = gpuBuffer.h_data;
    sortToCPUOrder(src, sorted, ids, state->gpd.idToIdxsOnCopy);

    Group &thisGroup = state->groups[lastGroupTag];

    ndf = thisGroup.getNDF();

    tempVector.erase(tempVector.begin(), tempVector.end());

    double conv = state->units.mvv_to_eng / state->units.boltz / 3.0;

    for (int i=0; i<src.size(); i++) {
        if (atoms[i].groupTag & lastGroupTag) {
            tempVector.push_back(sorted[i] * conv);
        }
    }
}

void DataComputerTemperature::computeTensor_CPU() {
    Virial total = *(Virial *) &gpuBuffer.h_data[0];
    total *= (state->units.mvv_to_eng / state->units.boltz);
    
    // just so that we have ndf, in case other routines reference it
    Group &thisGroup = state->groups[lastGroupTag];
    ndf = thisGroup.getNDF();
    /*
       int n;
       if (lastGroupTag == 1) {
       n = state->atoms.size();
       } else {
       n = * (int *) &gpuBuffer.h_data[1];
       }
     */
    tempTensor = total;
}

void DataComputerTemperature::computeTensorFromScalar() {
    int zeroDim = 3;
    if (state->is2d) {
        zeroDim = 2;
        tempTensor[0] = tempTensor[1] = totalKEScalar / 2.0;
    } else {
        tempTensor[0] = tempTensor[1] = tempTensor[2] = totalKEScalar / 3.0;
    }
    for (int i=zeroDim; i<6; i++) {
        tempTensor[i] = 0;
    }

}

void DataComputerTemperature::computeScalarFromTensor() {
    //int n;
    /*
    if (lastGroupTag == 1) {
        n = state->atoms.size();//\* (int *) &gpuBuffer.h_data[1];
    } else {
        n = * (int *) &gpuBuffer.h_data[1];
    }
    */ //
    /*
    if (state->is2d) {
        ndf = 2*(n-1); //-1 is analagous to extra_dof in lammps
    } else {
        ndf = 3*(n-1);
    }
    */
    Group &thisGroup = state->groups[lastGroupTag];
    ndf = thisGroup.getNDF();
    //ndf = state->groups[lastGroupTag].getNDF();
    totalKEScalar = (tempTensor[0] + tempTensor[1] + tempTensor[2]) * state->units.boltz;
    tempScalar = totalKEScalar / ndf;


}

void DataComputerTemperature::appendScalar(boost::python::list &vals) {
    vals.append(tempScalar);
}
void DataComputerTemperature::appendVector(boost::python::list &vals) {
    vals.append(tempVector);
}
void DataComputerTemperature::appendTensor(boost::python::list &vals) {
    vals.append(tempTensor);
}


