#include "IntegratorLGJF.h"

#include <chrono>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "Logging.h"
#include "State.h"

namespace py = boost::python;

// todo: add alpha to preForce_cu call; add alpha as parameter within constructor 
__global__ void preForce_cu_lang(int nAtoms, float4 *xs, float4 *vs, float4 *fs,
                            curandState_t *randStates, float alpha, float scaleFactor,
                            float dt)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
       
        float4 vel = vs[idx];
        float invmass = vel.w;
        // todo: examine this algorithm for numerical stability; optimize for floating point
        // operations with disparate magnitudes

        // friction coefficient alpha, used to calculate b param for a given atom/molecule
        
        float denominator_val = 1.0f + ( 0.5f *  ( alpha * dt * invmass) );
        float b_param = 1.0f / (denominator_val);
        //printf("denominator_val %f\n", denominator_val);
        curandState_t *randState = randStates + idx;
        curandState_t localState=*randState;

        // beta, a normally distributed random number used to move the step forward.
        // applied to both x position and velocity for a given particle.
        float3 beta ;
        beta.x = curand_normal(&localState) * scaleFactor;
        beta.y = curand_normal(&localState) * scaleFactor;
        beta.z = curand_normal(&localState) * scaleFactor;
        
        //printf("beta %f  %f  %f\n", beta.x, beta.y, beta.z);
        
        *randState = localState;
        float4 force = fs[idx];
        float4 pos = xs[idx];
        //printf("vel %f %f %f\n", vel.x, vel.y, vel.z);
        // we now have or vel, pos, and forces extracted from vs, xs, and fs, respectively
        //printf("force %f %f %f\n", force.x, force.y, force.z);
        //printf("pos %f %f %f\n", pos.x, pos.y, pos.z);
        // first, update dx with current velocity & force information
        float3 dx = make_float3(0.0f, 0.0f, 0.0f);
        float3 forceToPrint = make_float3(force);
        float3 velToPrint = make_float3(vel);

        //printf("make_float3(vel) : %f %f %f\n", velToPrint.x, velToPrint.y, velToPrint.z);
        //printf("make_Float3(force) : %f %f %f\n", forceToPrint.x, forceToPrint.y, forceToPrint.z);
        //printf("dx init: %f %f %f\n", dx.x, dx.y, dx.z);
        //printf("b_param dt velx vely velz: %f %f %f %f %f\n", b_param, dt, velToPrint.x, velToPrint.y, velToPrint.z);
        dx += (b_param * dt * make_float3(vel));
        //printf("dx 1 : %f %f %f\n", dx.x, dx.y, dx.z);
        dx += (0.5f * b_param * dt * dt * invmass * make_float3(force));
        //printf("dx 2 : %f %f %f\n", dx.x, dx.y, dx.z);
        dx += (0.5f * b_param * dt * invmass * beta);
        
        //printf("dx final %f %f %f\n", dx.x, dx.y, dx.z);


        // update dv with the information we currently have (still need x_n+1 and f_n+1,
        // to be computed in postForce)
        float3 dv = make_float3(0.0f, 0.0f, 0.0f);
        dv += (0.5f * dt * invmass * make_float3(force) );
        dv += (alpha * invmass * make_float3(pos));
        dv += (invmass * beta);
        //printf("dv %f %f %f\n", dv.x, dv.y, dv.z);


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
// todo: double check pre- and postForce_cu to verify this algorithm matches the literature
__global__ void postForce_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs, float alpha, float dt)
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
IntegratorLGJF::IntegratorLGJF(State *state_, double temp_, double alpha_, int seed_) : 
        Integrator(state_) , temp(temp_), alpha(alpha_), seed(seed_)
{
 // assert statement here if you can think of one that is applicable
}
// todo: various temperature constructors compatible with the interpolator
void __global__ initRandStatesLGJF(int nAtoms, curandState_t *states, int seed,int turn) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        curand_init(seed, idx, turn, states + idx);
    }

}

// 
void IntegratorLGJF::run(int numTurns)
{

    basicPreRunChecks();
    basicPrepare(numTurns);

    int periodicInterval = state->periodicInterval;

    auto start = std::chrono::high_resolution_clock::now();
    bool computeVirialsInForce = state->dataManager.computeVirialsInForce;

    randStates = GPUArrayDeviceGlobal<curandState_t>(state->atoms.size());
    initRandStatesLGJF<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), randStates.data(), seed,state->turn);
    
    float thisdt = state->dt;
    float scaleFactor = sqrt(2.0f * alpha * thisdt * temp);
    printf("scaleFactor %f \n", scaleFactor);

    for (int i=0; i<numTurns; ++i) {
        if (state->turn % periodicInterval == 0) {
            state->gridGPU.periodicBoundaryConditions();
        }
        // Prepare for timestep
        //! \todo Should asyncOperations() and doDataCollection() go into
        //!       Integrator::stepInit()? Same for periodicBoundayConditions()
        asyncOperations();
        doDataComputation();
        printf("Turn %d still going strong\n", i);
        stepInit(computeVirialsInForce);
        // initialize the random states for this turn, to be used in preForce
        // Perform first half of G-JF Langevin integration
        preForce();

        // Recalculate forces
        force(computeVirialsInForce);

        // Perform second half of G-JF Langevin integration (update velocities)
        postForce();

        stepFinal();
        doDataAppending();
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
    preForce_cu_lang<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.xs.getDevData(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            randStates.data(),
            alpha,
            scaleFactor,
            state->dt);
}

void IntegratorLGJF::postForce()
{
    uint activeIdx = state->gpd.activeIdx();
    postForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.xs.getDevData(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            alpha,
            state->dt);
}

void export_IntegratorLGJF()
{
    py::class_<IntegratorLGJF, boost::shared_ptr<IntegratorLGJF>,
               py::bases<Integrator>, boost::noncopyable>
    (
        "IntegratorLGJF",
        py::init<State *, double, double, int >(
            py::args("state", "temp", "alpha", "seed")
        )
    )

    .def("run", &IntegratorLGJF::run)
    /*
    .def_readwrite("alpha", &IntegratorLGJF::alpha)
    .def_readwrite("seed", &IntegratorLGJF::seed)
    */
    ;
}
