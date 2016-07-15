#include "IntegratorLGJF.h"

#include <chrono>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "Logging.h"
#include "State.h"

namespace py = boost::python;

// TODO: add alpha to preForce_cu call; add alpha as parameter within constructor / state etc.
__global__ void preForce_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs,
                            float alpha, float dt)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
       
        float4 vel = vs[idx];
        float invmass = vel.w;
        // TODO: examine this algorithm for numerical stability; optimize for floating point
        // operations with disparate magnitudes

        // friction coefficient alpha, used to calculate b param for a given atom/molecule
        // TODO: allow friction coefficient to be a function(x,y,z,v.x,v.y,v.z) etc.?
        // if so, incorporate as inline function called from here, possibly an evaluator class
        // or just an object of the state.  Allow alpha to be a function or a scalar constant.
        // actually, this method is only correct if alpha is a constant and NOT a function of position
        // as that introduces ambiguity.  Or does it? Idk, need more literature.
        float denominator_val = 1.0f + ( 0.5f *  ( alpha * dt * invmass) );
        float b_param = 1.0f / (denominator_val);

        // TODO: Un-pseudocode this
        // also, make it type float3, as that is how it will be used!
        float3 beta = CALL_GAUSSIAN_RNG();
        // end pseudocode

        float4 force = fs[idx];
        float4 pos = xs[idx];

        // we now have or vel, pos, and forces extracted from vs, xs, and fs, respectively

        // first, update dx with current velocity & force information
        float3 dx = (b_param * dt * vel);
        dx += (0.5f * b_param * dt * dt * invmass * make_float3(force));
        dx += (0.5f * b_param * dt * invmass * beta);

        // update dv with the information we currently have (still need x_n+1 and f_n+1,
        // to be computed in postForce)
        float3 dv = (0.5f * dt * invmass * make_float3(force) );
        dv += (alpha * invmass * make_float3(pos));
        dv += (invmass * beta);

        // now, add the changes to pos and vel, and set xs and vs to pos and vel, respectively
        pos += dx;
        vel += dv;

        xs[idx] = pos;
        vs[idx] = vel;

        // in order to move forward and complete the process of updating the velocity,
        // we will need to compute the forces at x_n+1 (the new xs[idx])
        fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);

    }
}
// TODO: double check pre- and postForce_cu to verify this algorithm matches the literature
__global__ void postForce_cu(int nAtoms, float4 *vs, float4 *fs, float alpha, float dt)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // we now have f_n+1, x_n+1; complete updating the velocity
        float4 vel = vs[idx];
        float invmass = vel.w;
        float4 pos = xs[idx];
        float4 force = fs[idx];
        
        float3 dv = 0.5f * dt * invmass * make_float3(force);
        dv += (-1.0f * alpha  * invmass * make_float3(pos));
        vel += dv;
        vs[idx] = vel;

        // ok; done. vs, xs, fs now in sync
    }
}


// 
IntegratorLGJF::IntegratorLGJF(State *state_)
    : Integrator(state_)
{

}
void IntegratorLGJF::run(int numTurns)
{

    basicPreRunChecks();
    basicPrepare(numTurns);

    int periodicInterval = state->periodicInterval;

    auto start = std::chrono::high_resolution_clock::now();
    bool computeVirials = state->computeVirials;
    for (int i=0; i<numTurns; ++i) {
        if (state->turn % periodicInterval == 0) {
            state->gridGPU.periodicBoundaryConditions();
        }
        // Prepare for timestep
        //! \todo Should asyncOperations() and doDataCollection() go into
        //!       Integrator::stepInit()? Same for periodicBoundayConditions()
        asyncOperations();
        doDataCollection();

        stepInit(computeVirials);

        // Perform first half of G-JF Langevin integration
        preForce();

        // Recalculate forces
        force(computeVirials);

        // Perform second half of G-JF Langevin integration (update velocities)
        postForce();

        stepFinal();

        //! \todo The following parts could also be moved into stepFinal
        state->turn++;
        if (state->verbose && (i+1 == numTurns || state->turn % state->shoutEvery == 0)) {
            mdMessage("Turn %d %.2f percent done.\n", (int)state->turn, 100.0*(i+1)/numTurns);
        }
    }

    //! \todo These parts could be moved to basicFinish()
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    mdMessage("runtime %f\n%e particle timesteps per second\n",
              duration.count(), state->atoms.size()*numTurns / duration.count());

    basicFinish();
}

void IntegratorLGJF::preForce()
{
    uint activeIdx = state->gpd.activeIdx();
    preForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.xs.getDevData(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            state->alpha,
            state->dt);
}

void IntegratorLGJF::postForce()
{
    uint activeIdx = state->gpd.activeIdx();
    postForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            state->alpha,
            state->dt);
}

void export_IntegratorLGJF()
{
    py::class_<IntegratorLGJF,
               boost::shared_ptr<IntegratorLGJF>,
               py::bases<Integrator>,
               boost::noncopyable>
    (
        "IntegratorLGJF",
        py::init<State *>()
    )
    .def("run", &IntegratorLGJF::run)
    ;
}
