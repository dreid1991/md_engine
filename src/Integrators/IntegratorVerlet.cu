#include "IntegratorVerlet.h"

#include <chrono>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "Logging.h"
#include "State.h"
#include "Fix.h"
using namespace MD_ENGINE;

namespace py = boost::python;

__global__ void nve_v_cu(int nAtoms, float4 *vs, float4 *fs, float dtf) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = dtf * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;
        fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
    }
}

__global__ void nve_x_cu(int nAtoms, float4 *xs, float4 *vs, float dt) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update position by a full timestep
        float4 vel = vs[idx];
        float4 pos = xs[idx];

        //printf("pos %f %f %f\n", pos.x, pos.y, pos.z);
        //printf("vel %f %f %f\n", vel.x, vel.y, vel.z);
        float3 dx = dt*make_float3(vel);
        pos += dx;
        xs[idx] = pos;
    }
}

__global__ void nve_xPIMD_cu(int nRingPoly, int nPerRingPoly, float omegaP, float4 *xs, float4 *vs, float dt) {

    // Declare relevant variables for NM transformation
    int idx = GETIDX();
    extern __shared__ float3 xsvs[];
    float3 *xsNM = xsvs;					// normal-mode transform of position
    float3 *vsNM = xsvs + PERBLOCK * nPerRingPoly;		// normal-mode transform of velocity

    if (idx < nRingPoly) {
	int baseIdx = idx * nPerRingPoly;
	
        // 1. Transform to normal mode positions and velocities
	// 	xNM_k = \sum_{n=1}^P x_n* Cnk
	// 	Cnk = \sqrt(1/P)			k = 0
	// 	Cnk = \sqrt(2/P) cos(2*pi*k*n/P)	1<= k <= P/2 -1
	// 	Cnk = \sqrt(1/P)(-1)^n			k = P/2
	// 	Cnk = \sqrt(2/P) sin(2*pi*k*n/P)	P/2+1<= k <= P -1
	// 2. advance positions/velocities by full timestep according
	// to free ring-polymer evolution
	// 3. back transform to regular coordinates
	float invP            = 1.0 / (float) nPerRingPoly;
	float twoPiInvP       = 2.0f * M_PI * invP;
	float invSqrtP 	      = sqrt(invP);
	float sqrt2           = sqrt(2.0);
	int   halfP           = nPerRingPoly / 2;	// P must be even for the following transformation!!!

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// 1. COORDINATE TRANSFORMATION
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// k = 0, n = 1,...,P
	for (int n = 0; n < nPerRingPoly; n++) {
	  xsNM[0] += make_float3(xs[baseIdx+n]);
	  vsNM[0] += make_float3(vs[baseIdx+n]);
	}
	// k = P/2, n = 1,...,P
	for (int n = 2; n < nPerRingPoly+1; n+=2) {
	  xsNM[halfP] -= make_float3(xs[baseIdx+n-2]);
	  xsNM[halfP] += make_float3(vs[baseIdx+n-1]);
	}
	// k = 1,...,P/2-1; n = 1,...,P
        for (int k = 1; k < halfP; k++) {
	  for (int n = 1; n < nPerRingPoly+1; n++) {
	   float3 xn     = make_float3(xs[baseIdx+n-1]);
	   float3 vn     = make_float3(vs[baseIdx+n-1]);
	   float  cosval = cos(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
	   xsNM[k] += xn * sqrt2 * cosval;
	   vsNM[k] += vn * sqrt2 * cosval;
	  }
	}
	// k = P/2+1,...,P-1; n = 1,...,P
        for (int k = halfP+1; k < nPerRingPoly; k++) {
	  for (int n = 1; n < nPerRingPoly+1; n++) {
	   float3 xn     = make_float3(xs[baseIdx+n-1]);
	   float3 vn     = make_float3(vs[baseIdx+n-1]);
	   float  sinval = sin(twoPiInvP * k * n);	// sin(2*pi*k*n/P)
	   xsNM[k] += xn * sqrt2 * sinval;
	   vsNM[k] += vn * sqrt2 * sinval;
	  }
	}
	// orthogonalize
	for (int n = 0; n<nPerRingPoly; n++) {
          xsNM[0] *= invSqrtP;
          vsNM[0] *= invSqrtP;
	}

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// 2. NORMAL-MODE RP COORDINATE EVOLUTION
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// xk(t+dt) = xk(t)*cos(om_k*dt) + vk(t)*sin(om_k*dt)/om_k
	// vk(t+dt) = vk(t)*cos(om_k*dt) - xk(t)*sin(om_k*dt)*om_k
	// k = 0
        xsNM[0] += vsNM[0] * dt; 
	// k = 1,...,P-1
	for (int k = 1; k< nPerRingPoly; k++) {
	  float omegaK = 2.0f * omegaP * sin( k * twoPiInvP * 0.5);
	  float cosdt  = cos(omegaK * dt);
	  float sindt  = sin(omegaK * dt);
	  float3 xsNMk = xsNM[k];
	  float3 vsNMk = vsNM[k];
	  xsNM[k] *= cosdt;
	  vsNM[k] *= cosdt;
	  xsNM[k] += vsNMk * sindt / omegaK;
	  vsNM[k] -= xsNMk * sindt * omegaK;
	}

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// 3. COORDINATE BACK TRANSFORMATION
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	for (int n = 1; n < nPerRingPoly+1; n++) {
	  // k = 0
	  float3 xn = xsNM[0]; 
	  float3 vn = vsNM[0]; 

	  // k = halfP
	  // xn += xsNM[halfP]*(-1)**n
	  if ( n % 2 == 0) {
	    xn += xsNM[halfP];
	    vn += vsNM[halfP];}
	  else {
	    xn -= xsNM[halfP];
	    vn -= vsNM[halfP];}

	  // k = 1,...,P/2-1; n = 1,...,P
	  for (int k = 1; k < halfP; k++) {
	    float  cosval = cos(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
	    xn += xsNM[k] * sqrt2 * cosval;
	    vn += vsNM[k] * sqrt2 * cosval;
	  }

	  // k = P/2+1,...,P-1; n = 1,...,P
          for (int k = halfP+1; k < nPerRingPoly; k++) {
	    float  sinval = sin(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
	    xn += xsNM[k] * sqrt2 * sinval;
	    vn += vsNM[k] * sqrt2 * sinval;
	  }

	  // replace evolved back-transformation
	  xs[baseIdx+n-1] = make_float4(xn*invSqrtP);
	  vs[baseIdx+n-1] = make_float4(vn*invSqrtP);
	}
    }

}
//so preForce_cu is split into two steps (nve_v, nve_x) if any of the fixes (barostat, for example), need to throw a step in there (as determined by requiresPostNVE_V flag)
__global__ void preForce_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs,
                            float dt, float dtf)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = dtf * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;

        // Update position by a full timestep
        float4 pos = xs[idx];

        //printf("vel %f %f %f\n", vel.x, vel.y, vel.z);
        float3 dx = dt*make_float3(vel);
        pos += dx;
        xs[idx] = pos;

        // Set forces to zero before force calculation
        fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
    }
}

// alternative version of preForce_cu which allows for normal-mode propagation of RP dynamics
// need to pass nPerRingPoly and omega_w
__global__ void preForcePIMD_cu(int nRingPoly, int nPerRingPoly, float omegaP, float4 *xs, float4 *vs, float4 *fs,
                            float dt, float dtf)
{
    // Declare relevant variables for NM transformation
    int idx = GETIDX();
    extern __shared__ float3 xsvs[];
    float3 *xsNM = xsvs;					// normal-mode transform of position
    float3 *vsNM = xsvs + PERBLOCK * nPerRingPoly;		// normal-mode transform of velocity

    if (idx < nRingPoly) {
	
        // Update velocity by a half timestep for all beads in the ring polymer
	int baseIdx = idx * nPerRingPoly;
	for (int id = baseIdx; id< baseIdx + nPerRingPoly; id++) {
        	float4 vel     = vs[id];
        	float  invmass = vel.w;
        	float4 force   = fs[id];
        	float3 dv      = dtf * invmass * make_float3(force);
        	vel           += dv;
        	vs[id]         = vel;
        	// Set forces to zero before force calculation
		fs[id] = make_float4(0.0f,0.0f,0.0f,force.w);
	}

        // 1. Transform to normal mode positions and velocities
	// 	xNM_k = \sum_{n=1}^P x_n* Cnk
	// 	Cnk = \sqrt(1/P)			k = 0
	// 	Cnk = \sqrt(2/P) cos(2*pi*k*n/P)	1<= k <= P/2 -1
	// 	Cnk = \sqrt(1/P)(-1)^n			k = P/2
	// 	Cnk = \sqrt(2/P) sin(2*pi*k*n/P)	P/2+1<= k <= P -1
	// 2. advance positions/velocities by full timestep according
	// to free ring-polymer evolution
	// 3. back transform to regular coordinates
	float invP            = 1.0 / (float) nPerRingPoly;
	float twoPiInvP       = 2.0f * M_PI * invP;
	float invSqrtP 	      = sqrt(invP);
	float sqrt2           = sqrt(2.0);
	int   halfP           = nPerRingPoly / 2;	// P must be even for the following transformation!!!

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// 1. COORDINATE TRANSFORMATION
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// k = 0, n = 1,...,P
	for (int n = 0; n < nPerRingPoly; n++) {
	  xsNM[0] += make_float3(xs[baseIdx+n]);
	  vsNM[0] += make_float3(vs[baseIdx+n]);
	}
	// k = P/2, n = 1,...,P
	for (int n = 2; n < nPerRingPoly+1; n+=2) {
	  xsNM[halfP] -= make_float3(xs[baseIdx+n-2]);
	  xsNM[halfP] += make_float3(vs[baseIdx+n-1]);
	}
	// k = 1,...,P/2-1; n = 1,...,P
        for (int k = 1; k < halfP; k++) {
	  for (int n = 1; n < nPerRingPoly+1; n++) {
	   float3 xn     = make_float3(xs[baseIdx+n-1]);
	   float3 vn     = make_float3(vs[baseIdx+n-1]);
	   float  cosval = cos(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
	   xsNM[k] += xn * sqrt2 * cosval;
	   vsNM[k] += vn * sqrt2 * cosval;
	  }
	}
	// k = P/2+1,...,P-1; n = 1,...,P
        for (int k = halfP+1; k < nPerRingPoly; k++) {
	  for (int n = 1; n < nPerRingPoly+1; n++) {
	   float3 xn     = make_float3(xs[baseIdx+n-1]);
	   float3 vn     = make_float3(vs[baseIdx+n-1]);
	   float  sinval = sin(twoPiInvP * k * n);	// sin(2*pi*k*n/P)
	   xsNM[k] += xn * sqrt2 * sinval;
	   vsNM[k] += vn * sqrt2 * sinval;
	  }
	}
	// orthogonalize
	for (int n = 0; n<nPerRingPoly; n++) {
          xsNM[0] *= invSqrtP;
          vsNM[0] *= invSqrtP;
	}

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// 2. NORMAL-MODE RP COORDINATE EVOLUTION
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// xk(t+dt) = xk(t)*cos(om_k*dt) + vk(t)*sin(om_k*dt)/om_k
	// vk(t+dt) = vk(t)*cos(om_k*dt) - xk(t)*sin(om_k*dt)*om_k
	// k = 0
        xsNM[0] += vsNM[0] * dt; 
	// k = 1,...,P-1
	for (int k = 1; k< nPerRingPoly; k++) {
	  float omegaK = 2.0f * omegaP * sin( k * twoPiInvP * 0.5);
	  float cosdt  = cos(omegaK * dt);
	  float sindt  = sin(omegaK * dt);
	  float3 xsNMk = xsNM[k];
	  float3 vsNMk = vsNM[k];
	  xsNM[k] *= cosdt;
	  vsNM[k] *= cosdt;
	  xsNM[k] += vsNMk * sindt / omegaK;
	  vsNM[k] -= xsNMk * sindt * omegaK;
	}

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// 3. COORDINATE BACK TRANSFORMATION
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	for (int n = 1; n < nPerRingPoly+1; n++) {
	  // k = 0
	  float3 xn = xsNM[0]; 
	  float3 vn = vsNM[0]; 

	  // k = halfP
	  // xn += xsNM[halfP]*(-1)**n
	  if ( n % 2 == 0) {
	    xn += xsNM[halfP];
	    vn += vsNM[halfP];}
	  else {
	    xn -= xsNM[halfP];
	    vn -= vsNM[halfP];}

	  // k = 1,...,P/2-1; n = 1,...,P
	  for (int k = 1; k < halfP; k++) {
	    float  cosval = cos(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
	    xn += xsNM[k] * sqrt2 * cosval;
	    vn += vsNM[k] * sqrt2 * cosval;
	  }

	  // k = P/2+1,...,P-1; n = 1,...,P
          for (int k = halfP+1; k < nPerRingPoly; k++) {
	    float  sinval = sin(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
	    xn += xsNM[k] * sqrt2 * sinval;
	    vn += vsNM[k] * sqrt2 * sinval;
	  }

	  // replace evolved back-transformation
	  xs[baseIdx+n-1] = make_float4(xn*invSqrtP);
	  vs[baseIdx+n-1] = make_float4(vn*invSqrtP);
	}
	
    }
}

__global__ void postForce_cu(int nAtoms, float4 *vs, float4 *fs, float dtf)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocities by a halftimestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = dtf * invmass * make_float3(force);
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
    basicPrepare(numTurns); //nlist built here
    force(false);
    int periodicInterval = state->periodicInterval;

	
    auto start = std::chrono::high_resolution_clock::now();
    DataManager &dataManager = state->dataManager;
    dtf = 0.5 * state->dt * state->units.ftm_to_v;
    for (int i=0; i<numTurns; ++i) {
        if (state->turn % periodicInterval == 0) {
            state->gridGPU.periodicBoundaryConditions();
        }
        bool computeVirialsInForce = dataManager.virialTurns.find(state->turn) != dataManager.virialTurns.end();

        stepInit(computeVirialsInForce);

        // Perform first half of velocity-Verlet step
        if (state->requiresPostNVE_V) {
            nve_v();
            postNVE_V();
            nve_x();
        } else {
            preForce();
        }
        postNVE_X();
        //printf("preForce IS COMMENTED OUT\n");

        handleBoundsChange();

        // Recalculate forces
        force(computeVirialsInForce);


        // Perform second half of velocity-Verlet step
        postForce();

        stepFinal();

        asyncOperations();
        //HEY - MAKE DATA APPENDING HAPPEN WHILE SOMETHING IS GOING ON THE GPU.  
        doDataComputation();
        doDataAppending();
        dataManager.clearVirialTurn(state->turn);

        //! \todo The following parts could also be moved into stepFinal
        state->turn++;
        if (state->verbose && (i+1 == numTurns || state->turn % state->shoutEvery == 0)) {
            mdMessage("Turn %d %.2f percent done.\n", (int)state->turn, 100.0*(i+1)/numTurns);
        }
    }

    //! \todo These parts could be moved to basicFinish()
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("after run\n");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    mdMessage("runtime %f\n%e particle timesteps per second\n",
              duration.count(), state->atoms.size()*numTurns / duration.count());

    basicFinish();
}

void IntegratorVerlet::nve_v() {
    uint activeIdx = state->gpd.activeIdx();
    nve_v_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            dtf);
}

void IntegratorVerlet::nve_x() {
    uint activeIdx = state->gpd.activeIdx();
    if (state->nPerRingPoly == 1) {
    	nve_x_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
    	        state->atoms.size(),
    	        state->gpd.xs.getDevData(),
    	        state->gpd.vs.getDevData(),
    	        state->dt); }
    else {
	// get target temperature from thermostat fix
	double temp;
	for (Fix *f: state->fixes) {
	  if ( f->isThermostat && f->groupHandle == "all" ) {
	    std::string t = "temp";
	    temp = f->getInterpolator(t)->getCurrentVal();
	  }
	}
	int   nPerRingPoly = state->nPerRingPoly;
    int   nRingPoly = state->atoms.size() / nPerRingPoly;
	float omegaP    = (float) nPerRingPoly / state->units.hbar * state->units.boltz * temp;
    	nve_xPIMD_cu<<<NBLOCK(nRingPoly), PERBLOCK, sizeof(float3) * 2 *PERBLOCK * nPerRingPoly>>>(
	        nRingPoly,
	 	nPerRingPoly,
		omegaP,
    	        state->gpd.xs.getDevData(),
    	        state->gpd.vs.getDevData(),
    	        state->dt); }
}
void IntegratorVerlet::preForce()
{
    uint activeIdx = state->gpd.activeIdx();
    if (state->nPerRingPoly == 1) {
    	preForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
    	        state->atoms.size(),
    	        state->gpd.xs.getDevData(),
    	        state->gpd.vs.getDevData(),
    	        state->gpd.fs.getDevData(),
    	        state->dt,
    	        dtf); }
    else {
	// get target temperature from thermostat fix
	// XXX: need to think about how to handle if no thermostat
	//      probably should not be allowed, tbh
	double temp;
	for (Fix *f: state->fixes) {
	  if ( f->isThermostat && f->groupHandle == "all" ) {
	    std::string t = "temp";
	    temp = f->getInterpolator(t)->getCurrentVal();
	  }
	}
	int   nPerRingPoly = state->nPerRingPoly;
        int   nRingPoly = state->atoms.size() / nPerRingPoly;
	float omegaP    = (float) nPerRingPoly / state->units.hbar * state->units.boltz * temp;
    	preForcePIMD_cu<<<NBLOCK(nRingPoly), PERBLOCK, sizeof(float3) * 2 *PERBLOCK * nPerRingPoly>>>(
	        nRingPoly,
	 	nPerRingPoly,
		omegaP,
    	        state->gpd.xs.getDevData(),
    	        state->gpd.vs.getDevData(),
    	        state->gpd.fs.getDevData(),
    	        state->dt,
    	        dtf); }
}

void IntegratorVerlet::postForce()
{
    uint activeIdx = state->gpd.activeIdx();
    postForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            dtf);
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
