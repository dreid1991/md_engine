#include "IntegraterVerlet.h"

#include "State.h"

IntegraterVerlet::IntegraterVerlet(SHARED(State) state_) : Integrater(state_.get(), IntVerletType) {
}

__global__ void preForce_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs, float4 *fsLast, float dt) {
    int idx = GETIDX();
    if (idx < nAtoms) {

        float4 vel = vs[idx];
        float4 force = fs[idx];

        float invmass = vel.w;
        float groupTag = force.w;
        //float id = pos.w;
        float4 dPos = vel * dt + force * dt*dt*0.5f*invmass;
        
        int zero = 0;
        dPos.w = * (int *) &zero;
        xs[idx] += dPos; //does this do single 16 byte transfer?

        //xs[idx] = pos;
        fsLast[idx] = force;
        fs[idx] = make_float4(0, 0, 0, groupTag);
    }
}


__global__ void postForce_cu(int nAtoms, float4 *vs, float4 *fs, float4 *fsLast, float dt) {
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
void IntegraterVerlet::preForce(uint activeIdx) {
	//vector<Atom> &atoms = state->atoms;
    preForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.xs.getDevData(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(), state->gpd.fsLast.getDevData(), state->dt);
}


void IntegraterVerlet::postForce(uint activeIdx) {
    postForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(), state->gpd.fsLast.getDevData(), state->dt);
}



void IntegraterVerlet::run(int numTurns) {
    basicPreRunChecks(); 
    basicPrepare(numTurns);
    
    double rCut = state->rCut;
    double padding = state->padding;

    //ADD PADDING


    int periodicInterval = state->periodicInterval;
    int numBlocks = ceil(state->atoms.size() / (float) PERBLOCK);
    int remainder = state->turn % periodicInterval;
    int turnInit = state->turn; 
    auto start = std::chrono::high_resolution_clock::now();
    /*
    int x = 0;
    dataGather = async(launch::async, [&]() {x=5;});
    cout << "valid " << endl;
    cout << dataGather.valid() << endl;
    dataGather.wait();
    cout << dataGather.valid() << endl;
    cout << "x is " << x << endl;
    cout << "waiting again" << endl;
    dataGather.wait();
    cout << "past" << endl;
    return;
     */
    for (int i=0; i<numTurns; i++) {
        if (! ((remainder + i) % periodicInterval)) {
            state->gridGPU.periodicBoundaryConditions(rCut + padding, true);
        }
        int activeIdx = state->gpd.activeIdx;
        asyncOperations();
        doDataCollection();
        preForce(activeIdx);
        force(activeIdx);
        postForce(activeIdx);

        if (state->verbose and not ((state->turn - turnInit) % state->shoutEvery)) {
            cout << "Turn " << (int) state->turn << " " << (int) (100 * (state->turn - turnInit) / (num) numTurns) << " percent done" << endl;
        }
        state->turn++;

    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    cout << "runtime " << duration.count() << endl;
    cout << (state->atoms.size() * numTurns / duration.count()) << " particle timesteps per  second " << endl;

    basicFinish();

}

void export_IntegraterVerlet() {
    class_<IntegraterVerlet, SHARED(IntegraterVerlet), bases<Integrater>, boost::noncopyable > ("IntegraterVerlet", init<SHARED(State)>())
        .def("run", &IntegraterVerlet::run)
        ;
}

