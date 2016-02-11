#include "IntegraterVerlet.h"



IntegraterVerlet::IntegraterVerlet(SHARED(State) state_) : Integrater(state_.get(), IntVerletType) {
}

__global__ void preForce_cu(int nAtoms, cudaSurfaceObject_t xs, float4 *vs, float4 *fs, float4 *fsLast, float dt) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        int xIdx = XIDX(idx, sizeof(float4));
        int yIdx = YIDX(idx, sizeof(float4));
        int xAddr = xIdx * sizeof(float4);
        float4 pos = surf2Dread<float4>(xs, xAddr, yIdx);

        float4 vel = vs[idx];
        float4 force = fs[idx];

        float invmass = vel.w;
        float groupTag = force.w;
        float id = pos.w;
        pos += vel * dt + force * dt*dt*0.5f*invmass;
        pos.w = id;

        surf2Dwrite(pos, xs, xAddr, yIdx);
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
    preForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.xs.getSurf(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(), state->gpd.fsLast.getDevData(), state->dt);
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

    for (int i=0; i<numTurns; i++) {
        if (! ((remainder + i) % periodicInterval)) {
            state->gridGPU.periodicBoundaryConditions(rCut + padding, true);
        }
        int activeIdx = state->gpd.activeIdx;
        asyncOperations();
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
    class_<IntegraterVerlet, SHARED(IntegraterVerlet), bases<Integrater> > ("IntegraterVerlet", init<SHARED(State)>())
        .def("run", &IntegraterVerlet::run)
        ;
}

