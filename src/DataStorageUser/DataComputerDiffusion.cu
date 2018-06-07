#include "DataComputerDiffusion.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
#include "Fix.h"
#include "Group.h"
#include "GPUData.h"
#include "GridGPU.h"

namespace py = boost::python;
using namespace MD_ENGINE;
const std::string computer_type_ = "diffusion";

namespace {

__global__ void periodicWrap(real4 *xs, int nAtoms, BoundsGPU bounds) {

    int idx = GETIDX();
    if (idx < nAtoms) {

        real4 pos = xs[idx];

        real id = pos.w;
        real3 trace = bounds.trace();
        real3 diffFromLo = make_real3(pos) - bounds.lo;
#ifdef DASH_DOUBLE
        real3 imgs = floor(diffFromLo / trace); //are unskewed at this point
#else 
        real3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
#endif
        pos -= make_real4(trace * imgs * bounds.periodic);
        pos.w = id;
        //if (not(pos.x==orig.x and pos.y==orig.y and pos.z==orig.z)) { //sigh
        if (imgs.x != 0 or imgs.y != 0 or imgs.z != 0) {
            xs[idx] = pos;
        }
    }

}

__global__ void storeInitialPositions(int nAtoms,
                                      const real4 *__restrict__ xs, 
                                      real4 * __restrict__ xs_init,
                                      const int* __restrict__ idToIdxs) {
    int id = GETIDX();
    if (id < nAtoms) {
        // xs is simulation data, organized by idx
        int idx = idToIdxs[id];
        real4 pos = xs[idx];
        // xs_init is DataComputerDiffusion data, organized by id
        xs_init[id] = pos;
    }
}

// computes displacement scalar for all atoms of selected species at a given time t
__global__ void computeDisplacementScalar(int nAtoms,
                                       int species_type,
                                       const real4 * __restrict__ xs,
                                       const real4 * __restrict__ xs_init,
                                       real4 * __restrict__ xs_recent,
                                       real4       * __restrict__ boxes,
                                       real  * __restrict__ diffusion_scalar,
                                       const int* __restrict__ idToIdxs,
                                       BoundsGPU bounds) {

    int id = GETIDX();
    if (id < nAtoms) {
        int idx = idToIdxs[id];
        real4 pos_whole = xs[idx];
        int type = __real_as_int(pos_whole.w);

        if (type == species_type) {

            real4 pos_init_whole       = xs_init[id];
            real4 pos_recent_whole     = xs_recent[id];
            real4 boxes_traveled_whole = boxes[id];

            real3 pos                  = make_real3(pos_whole);
            real3 pos_init             = make_real3(pos_init_whole);
            real3 pos_recent           = make_real3(pos_recent_whole);
            real3 boxes_traveled       = make_real3(boxes_traveled_whole);

            real3 box_displacement     = pos - pos_recent;

            real3 boxDimensions = bounds.rectComponents;
            // if it moved half the box length in a given direction, we can safely assume that it 
            // actually was wrapped around by PBC
            if (fabsf(box_displacement.x) > (0.5 * boxDimensions.x)) {
                // if displacement is positive by > 0.5 boxDim, then it was wrapped around to the left
                if (box_displacement.x > 0) {
                    boxes_traveled += make_real3(-1.0, 0.0, 0.0);
                } else {
                    boxes_traveled += make_real3(1.0, 0.0, 0.0);
                }
            }
            if (fabsf(box_displacement.y) > (0.5 * boxDimensions.y)) {
                // if displacement is positive by > 0.5 boxDim, then it was wrapped around to the left
                if (box_displacement.y > 0) {
                    boxes_traveled += make_real3(0.0, -1.0, 0.0);
                } else {
                    boxes_traveled += make_real3(0.0, 1.0, 0.0);
                }

            }
            if (fabsf(box_displacement.z) > (0.5 * boxDimensions.z)) {
                // if displacement is positive by > 0.5 boxDim, then it was wrapped around to the left
                if (box_displacement.z > 0) {
                    boxes_traveled += make_real3(0.0, 0.0, -1.0);
                } else {
                    boxes_traveled += make_real3(0.0, 0.0, 1.0);
                }
            }

            // ok, we now have our up-to-date boxes_traveled array.
            real3 unwrapped = boxes_traveled * boxDimensions;
            real3 unwrapped_vector = pos - pos_init + unwrapped;
            real dist_sqr = lengthSqr(unwrapped_vector);
            diffusion_scalar[id] = dist_sqr;
            
            // then store pos in xs_recent at the end.
            xs_recent[id] = pos_whole;
            // update our boxes
            boxes[id] = make_real4(boxes_traveled);
        }
    }
}

} // namespace {}



DataComputerDiffusion::DataComputerDiffusion(State *state_, std::string computeMode_,std::string species_) : DataComputer(state_, computeMode_, false,computer_type_), 
                               species(species_) 
{
}


void DataComputerDiffusion::prepareForRun() {
    DataComputer::prepareForRun();
    // we do need to allocate a few arrays here

    // re-set this..
    gpuBufferReduce = GPUArrayGlobal<real>(2);
    // ok, we have our initial positions array - just make this nAtoms.

    species_type  = state->atomParams.typeFromHandle(species);

    species_count = 0;

    for (auto atom : state->atoms) {
        if (atom.type == species_type) {
            species_count++;
        }
    }
    // we're also always going to do this as id, not idxs - so just pass things as idToIdxs at all times
    xs_init = GPUArrayDeviceGlobal<real4>(state->gpd.xs.size());
    xs_recent = GPUArrayDeviceGlobal<real4>(state->gpd.xs.size());
    boxes_traveled = GPUArrayDeviceGlobal<real4>(state->gpd.xs.size());
    diffusion_scalar = GPUArrayDeviceGlobal<real>(state->gpd.xs.size()); // array of all displacements @ some time t
    // this is aggregated immediately after, then reset

    diffusion_vector = std::vector<double> (); // empty; push back as we grow.


    xs_init.memset(0);
    xs_recent.memset(0);
    boxes_traveled.memset(0);
    diffusion_scalar.memset(0);

    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    // copy state->gpd.xs as id to xs_init.  this is not going to change through the course of a run.
    // at this point, everything should be wrapped inside the box anyways, so nothing to do on that front;
    int nAtoms = state->atoms.size();
    periodicWrap<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, state->boundsGPU);

    storeInitialPositions<<<NBLOCK(nAtoms),PERBLOCK>>>(nAtoms,
                                                       gpd.xs(activeIdx),
                                                       xs_init.data(),
                                                       gpd.idToIdxs.d_data.data());

    // apply the same to xs_recent...
    storeInitialPositions<<<NBLOCK(nAtoms),PERBLOCK>>>(nAtoms,
                                                       gpd.xs(activeIdx),
                                                       xs_recent.data(),
                                                       gpd.idToIdxs.d_data.data());

}


void DataComputerDiffusion::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    int nAtoms = state->atoms.size();
    
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();

    gpuBufferReduce.d_data.memset(0); 
    diffusion_scalar.memset(0);

    periodicWrap<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, state->boundsGPU);

    cudaDeviceSynchronize();

    computeDisplacementScalar<<<NBLOCK(nAtoms),PERBLOCK>>>(nAtoms,
                                                           species_type,
                                                           gpd.xs(activeIdx),
                                                           xs_init.data(),
                                                           xs_recent.data(),
                                                           boxes_traveled.data(),
                                                           diffusion_scalar.data(),
                                                           gpd.idToIdxs.d_data.data(),
                                                           state->boundsGPU);

     accumulate_gpu<real, real, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(real)>>>
        (gpuBufferReduce.getDevData(), diffusion_scalar.data(), nAtoms, state->devManager.prop.warpSize, SumSingle());
    
    gpuBufferReduce.dataToHost();
    cudaDeviceSynchronize();
}

void DataComputerDiffusion::computeVector_CPU() {
    
    // use different time origins..??
    double avg_sqr_displacement = gpuBufferReduce.h_data[0] /( (double) species_count);
    /*
    // ok, verified: this does as it should, so we are scaling by the correct 'time'
    if (!((state->turn - state->runInit) % 100)) {
        std::cout << "state->turn: " << state->turn << "; state->runInit: " << state->runInit << std::endl;
    }
    */
    // as A^2/fs..
    double diffusion_coeff_Ang_fs = avg_sqr_displacement / (6.0); 
    
    // convert to A^2/ps for literature comparision...
    //double conversion = 1000.0; // converting /fs to /ps
    // we want the slope of the tail, not normalizing by total time, otherwise it never converges.
    //double diffusion_coeff = conversion * diffusion_coeff_Ang_fs;
    double diffusion_coeff = diffusion_coeff_Ang_fs;
    diffusion_vector.push_back(diffusion_coeff);
}

void DataComputerDiffusion::postRun(boost::python::list &vals) {

    vals = boost::python::list {};
    
    for (size_t i  = 0; i < diffusion_vector.size(); i++ ) {
        double currentVal = (double) diffusion_vector[i];
        vals.append(currentVal);
    }
}

