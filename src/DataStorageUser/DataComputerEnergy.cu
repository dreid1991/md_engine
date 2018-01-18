#include "DataComputerEnergy.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;


__global__ void zero_outside_bounds(int nAtoms, float4 *xs, float *perParticleEng, BoundsGPU bounds, BoundsGPU localBounds) {
    int idx = GETIDX();

    if (idx < nAtoms) {
        float4 thispos = xs[idx];
        // get the wrapped position; if it is within the specified local bounds, do nothing;
        // else, set the per particle energy to zero
        
        // copied from GridGPU.cu...
        float3 trace = bounds.trace();
        float3 diffFromLo = make_float3(thispos) - bounds.lo;
        float3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
        thispos -= make_float4(trace * imgs * bounds.periodic);
        float3 pos = make_float3(thispos);
        
        // ok; no need to write this to the global array though;
        float3 lo = localBounds.lo;
        float3 hi = localBounds.lo + localBounds.rectComponents;

        float energy = perParticleEng[idx];
        // now, check if the image is within the bounds specified by localBounds
        if (pos.x < lo.x) energy = 0.0;
        if (pos.y < lo.y) energy = 0.0;
        if (pos.z < lo.z) energy = 0.0;

        if (pos.x > hi.x) energy = 0.0;
        if (pos.y > hi.y) energy = 0.0;
        if (pos.z > hi.z) energy = 0.0;

        // write to global data
        perParticleEng[idx] = energy;
    }

}



DataComputerEnergy::DataComputerEnergy(State *state_, py::list fixes_, std::string computeMode_, std::string groupHandleB_) : DataComputer(state_, computeMode_, false), groupHandleB(groupHandleB_) {

    groupTagB = state->groupTagFromHandle(groupHandleB);
    otherIsAll = groupHandleB == "all";
    // initialize some other quantities to 0 / false etc.
    checkWithinBounds = false;
    lo = Vector(0.0, 0.0, 0.0);
    hi = lo;
    localBounds = BoundsGPU(); // default ctor
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

// constructor when bounds are passed in
DataComputerEnergy::DataComputerEnergy(State *state_, py::list fixes_, std::string computeMode_, std::string groupHandleB_, Vector lo_, Vector hi_) : DataComputer(state_, computeMode_, false), groupHandleB(groupHandleB_), lo(lo_), hi(hi_) {

    groupTagB = state->groupTagFromHandle(groupHandleB);
    otherIsAll = groupHandleB == "all";
    float3 lo_float3 = lo.asFloat3();
    float3 hi_float3 = hi.asFloat3();
    checkWithinBounds = true;
    bool *periodic = state->periodic;
    localBounds = BoundsGPU(lo_float3, lo_float3 + hi_float3, make_float3((int) periodic[0], (int) periodic[1], (int) periodic[2]));
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
    int activeIdx = gpd.activeIdx();
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
    if (checkWithinBounds) {
        // we don't need to use the nlist here, just get the positions of the atom and zero the energy if 
        // it is outside of the specified bounds
        zero_outside_bounds<<<NBLOCK(nAtoms),PERBLOCK>>> (nAtoms, gpd.xs(activeIdx),gpuBuffer.getDevData(), state->boundsGPU,localBounds);
    
    }

    if (groupTag == 1 or !otherIsAll) { //if other isn't all, then only group-group energies got computed so need to sum them all up anyway.  If other is all then every eng gets computed so need to accumulate only things in group
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
    int activeIdx = gpd.activeIdx();
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
    if (checkWithinBounds) {
        // we don't need to use the nlist here, just get the positions of the atom and zero the energy if 
        // it is outside of the specified bounds (we consider the periodic wrapping here as well)
        zero_outside_bounds<<<NBLOCK(nAtoms),PERBLOCK>>> (nAtoms, gpd.xs(activeIdx),gpuBuffer.getDevData(), state->boundsGPU,localBounds);
    }
    if (transferToCPU) {
        gpuBuffer.dataToHost();
    }
}




void DataComputerEnergy::computeScalar_CPU() {
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

void DataComputerEnergy::computeVector_CPU() {
    //ids have already been transferred, look in doDataComputation in integUtil
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

