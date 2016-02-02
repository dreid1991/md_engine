#include "IntegraterLangevin.h"

#include <curand_kernel.h>
 #include <math.h>
      

IntegraterLangevin::IntegraterLangevin(SHARED(State) state_) : IntegraterVerlet(state_) {
}


__global__ void postForce_Langevin_cu(int nAtoms, float4 *vs, float4 *fs, float4 *fsLast, float dt,int timesteps) {
    int idx = GETIDX();
    if (idx < nAtoms) {
      
	curandState_t localState;
	curand_init(timesteps, idx, 0, &localState);
	float2 g2;
	float4 Wiener;
	g2=curand_normal2(&localState);
	Wiener.x=g2.x;
	Wiener.y=g2.y;
	g2=curand_normal2(&localState);
	Wiener.z=g2.x;
	
	
        float4 vel = vs[idx];
        float4 force = fs[idx];
        float4 forceLast = fsLast[idx];
        float invmass = vel.w;
        //printf("invmass is %f\n", invmass);
        //printf("dt is %f\n", dt);
        //printf("force is %f %f %f\n", force.x, force.y, force.z);
        float4 newVel = vel + (forceLast + force+sqrt(6.0/dt)*Wiener-vel) * dt * 0.5f * invmass; //sqrt(6.0*T*gamma/dt)*Wiener;
	
        //printf("vel is %f %f %f\n", newVel.x, newVel.y, newVel.z);
        newVel.w = invmass;
        vs[idx] = newVel;
    }
}


void IntegraterLangevin::postForce(uint activeIdx,int timesteps) {
    postForce_Langevin_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(), state->gpd.fsLast.getDevData(), state->dt,timesteps);
}



void IntegraterLangevin::run(int numTurns) {
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
        postForce(activeIdx,state->turn);

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

// void export_IntegraterVerlet() {
//     class_<IntegraterVerlet, SHARED(IntegraterVerlet), bases<Integrater> > ("IntegraterVerlet", init<SHARED(State)>())
//         .def("run", &IntegraterVerlet::run)
//         ;
// }

