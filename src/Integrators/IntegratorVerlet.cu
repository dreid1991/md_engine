#include "IntegratorVerlet.h"

#include <chrono>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "Logging.h"
#include "State.h"

namespace py = boost::python;

__global__ void preForce_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs,
                            float dt)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = 0.5f * dt * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;

        // Update position by a full timestep
        float4 pos = xs[idx];

        float3 dx = dt*make_float3(vel);
        pos += dx;
        xs[idx] = pos;

        // Set forces to zero before force calculation
        fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
    }
}

__global__ void postForce_cu(int nAtoms, float4 *vs, float4 *fs, float dt)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocities by a halftimestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = 0.5f * dt * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;
    }
}

IntegratorVerlet::IntegratorVerlet(State *state_)
    : Integrator(state_)
{

}
void IntegratorVerlet::run(int numTurns)
{

    basicPreRunChecks();
    basicPrepare(numTurns);

    int periodicInterval = state->periodicInterval;

    auto start = std::chrono::high_resolution_clock::now();
    bool computeVirialsInForce = state->dataManager.computeVirialsInForce;
    for (int i=0; i<numTurns; ++i) {
        if (state->turn % periodicInterval == 0) {
            state->gridGPU.periodicBoundaryConditions();
        }
        // Prepare for timestep
        //! \todo Should asyncOperations() and doDataCollection() go into
        //!       Integrator::stepInit()? Same for periodicBoundayConditions()

        stepInit(computeVirialsInForce);

        // Perform first half of velocity-Verlet step
        preForce();

        // Recalculate forces
        force(computeVirialsInForce);
        asyncOperations();
        doDataComputation();

        // Perform second half of velocity-Verlet step
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

void IntegratorVerlet::preForce()
{
    uint activeIdx = state->gpd.activeIdx();
    preForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.xs.getDevData(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            state->dt);
}

void IntegratorVerlet::postForce()
{
    uint activeIdx = state->gpd.activeIdx();
    postForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            state->dt);
}

void export_IntegratorVerlet()
{
    py::class_<IntegratorVerlet,
               boost::shared_ptr<IntegratorVerlet>,
               py::bases<Integrator>,
               boost::noncopyable>
    (
        "IntegratorVerlet",
        py::init<State *>()
    )
    .def("run", &IntegratorVerlet::run)
    ;
}
