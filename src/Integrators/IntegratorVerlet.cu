#include "IntegratorVerlet.h"

#include "State.h"

IntegratorVerlet::IntegratorVerlet(SHARED(State) state_)
    : Integrator(state_.get()) {}

__global__ void preForce_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs,
                            float4 *fsLast, float dt) {
    int idx = GETIDX();
    if (idx < nAtoms) {

        float4 vel = vs[idx];
        float4 force = fs[idx];

        float invmass = vel.w;
        float groupTag = force.w;

        float3 dPos = make_float3(vel) * dt +
                      make_float3(force) * dt*dt * 0.5f * invmass;

        // Only add float3 to xs and fs! (w entry is used as int or bitmask)
        //THIS IS NONTRIVIALLY FASTER THAN DOING +=.  Sped up whole sumilation by 1%
        float4 xCur = xs[idx];
        xCur += dPos;
        xs[idx] = xCur;

        //xs[idx] = pos;
        fsLast[idx] = force;
        fs[idx] = make_float4(0, 0, 0, groupTag);
    }
}

__global__ void postForce_cu(int nAtoms, float4 *vs, float4 *fs,
                             float4 *fsLast, float dt) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 vel = vs[idx];
        float4 force = fs[idx];
        float4 forceLast = fsLast[idx];
        float invmass = vel.w;
        //printf("invmass is %f\n", invmass);
        //printf("dt is %f\n", dt);
        //printf("force is %f %f %f\n", force.x, force.y, force.z);
        float4 newVel = vel + (forceLast + force) * dt * 0.5f * invmass;
        //printf("vel is %f %f %f\n", newVel.x, newVel.y, newVel.z);
        newVel.w = invmass;
        vs[idx] = newVel;
    }
}

void IntegratorVerlet::preForce(uint activeIdx) {
    //vector<Atom> &atoms = state->atoms;
    preForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.xs.getDevData(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            state->gpd.fsLast.getDevData(),
            state->dt);
}

void IntegratorVerlet::postForce(uint activeIdx) {
    postForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            state->gpd.fsLast.getDevData(),
            state->dt);
}


void IntegratorVerlet::run(int numTurns) {
    basicPreRunChecks();
    basicPrepare(numTurns);

    int periodicInterval = state->periodicInterval;
    int numBlocks = ceil(state->atoms.size() / (float) PERBLOCK);
    int remainder = state->turn % periodicInterval;
    int turnInit = state->turn;
    auto start = std::chrono::high_resolution_clock::now();
    /*
    int x = 0;
    dataGather = async(launch::async, [&]() {x=5;});
    std::cout << "valid " << std::endl;
    std::cout << dataGather.valid() << std::endl;
    dataGather.wait();
    std::cout << dataGather.valid() << std::endl;
    std::cout << "x is " << x << std::endl;
    std::cout << "waiting again" << std::endl;
    dataGather.wait();
    std::cout << "past" << std::endl;
    return;
     */
    for (int i=0; i<numTurns; i++) {
        if (! ((remainder + i) % periodicInterval)) {
            state->gridGPU.periodicBoundaryConditions();
        }
        int activeIdx = state->gpd.activeIdx();
        asyncOperations();
        doDataCollection();
        preForce(activeIdx);
        force(activeIdx);
        postForce(activeIdx);

        if (state->verbose and not ((state->turn - turnInit) % state->shoutEvery)) {
            std::cout << "Turn " << (int) state->turn
                      << " " << (int) (100 * (state->turn - turnInit) / (num) numTurns)
                      << " percent done" << std::endl;
        }
        state->turn++;

    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "runtime " << duration.count() << std::endl;
    std::cout << (state->atoms.size() * numTurns / duration.count())
              << " particle timesteps per  second " << std::endl;

    basicFinish();

}

void export_IntegratorVerlet() {
    boost::python::class_<IntegratorVerlet,
                          SHARED(IntegratorVerlet),
                          boost::python::bases<Integrator>,
                          boost::noncopyable > (
        "IntegratorVerlet",
        boost::python::init<SHARED(State)>()
     )
    .def("run", &IntegratorVerlet::run)
    ;
}

