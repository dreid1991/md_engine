#include "IntegratorVerlet.h"

#include <chrono>

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "Logging.h"
#include "State.h"
#include "Fix.h"
#include "cutils_func.h"
#include "globalDefs.h"
using namespace MD_ENGINE;

namespace py = boost::python;

__global__ void nve_v_cu(int nAtoms, float4 *vs, float4 *fs, float dtf) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        float4 vel = vs[idx];
        float invmass = vel.w;
        float4 force = fs[idx];
        if (invmass > INVMASSBOOL) {
            fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
            vs[idx] = make_float4(0.0f, 0.0f, 0.0f, invmass);
            return;
        }

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

__global__ void nve_xPIMD_cu(int nAtoms, int nPerRingPoly, float omegaP, float4 *xs, float4 *vs, float dt) {

    // Declare relevant variables for NM transformation
    int idx = GETIDX(); 
    extern __shared__ float3 xsvs[];
    float3 *xsNM = xsvs;				// normal-mode transform of position
    float3 *vsNM = xsvs + PERBLOCK;		// normal-mode transform of velocity
    float3 *tbr  = xsvs + 2*PERBLOCK;   // working array to place variables "to be reduced"
    bool useThread = idx < nAtoms;
    float3 xn = make_float3(0, 0, 0);
    float3 vn = make_float3(0, 0, 0);
    float xW;
    float vW;

    // helpful reference indices/identifiers
    bool   needSync= nPerRingPoly>warpSize;
    bool   amRoot  = (threadIdx.x % nPerRingPoly) == 0;
    int    rootIdx = (threadIdx.x / nPerRingPoly) * nPerRingPoly;
    int    beadIdx = idx % nPerRingPoly;
    int    n       = beadIdx + 1;
	
    // 1. Transform to normal mode positions and velocities
	// 	xNM_k = \sum_{n=1}^P x_n* Cnk
	// 	Cnk = \sqrt(1/P)			k = 0
	// 	Cnk = \sqrt(2/P) cosf(2*pi*k*n/P)	1<= k <= P/2 -1
	// 	Cnk = \sqrt(1/P)(-1)^n			k = P/2
	// 	Cnk = \sqrt(2/P) sinf(2*pi*k*n/P)	P/2+1<= k <= P -1
	// 2. advance positions/velocities by full timestep according
	// to free ring-polymer evolution
	// 3. back transform to regular coordinates
	float invP            = 1.0f / (float) nPerRingPoly;
	float twoPiInvP       = 2.0f * M_PI * invP;
	float invSqrtP 	      = sqrtf(invP);
	float sqrt2           = sqrtf(2.0f);
	int   halfP           = nPerRingPoly / 2;	// P must be even for the following transformation!!!

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 1. COORDINATE TRANSFORMATION
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // first we will compute the normal mode coordinates for the positions, and then the velocities
    // using an identical structure. Each thread (bead) will contribute to each mode index, and
    // the strategy will be too store the results of each bead for a given mode in a working array
    // reduce the result by summation and store that in the final array for generating the positions

    // %%%%%%%%%%% POSITIONS %%%%%%%%%%%
    if (useThread) {
        float4 xWhole = xs[idx];
        float4 vWhole = vs[idx];
        xn = make_float3(xWhole);
        vn = make_float3(vWhole);
        xW = xWhole.w;
        vW = vWhole.w;

        // if the particle is massless, nothing more to do
        if (vW > INVMASSBOOL) {
            vWhole = make_float4(0.0, 0.0, 0.0, vW);
            return;
        }
    }
    // k = 0, n = 1,...,P
    tbr[threadIdx.x]  = xn;
    if (needSync)    __syncthreads();
    reduceByN<float3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {xsNM[threadIdx.x] = tbr[threadIdx.x]*invSqrtP;}

    // k = P/2, n = 1,...,P
    if (threadIdx.x % 2 == 0) {
        tbr[threadIdx.x] = xn * -1;
    } else {
        tbr[threadIdx.x] = xn ;
    }
    if (needSync)   { __syncthreads();}
    reduceByN<float3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {xsNM[threadIdx.x+halfP] = tbr[threadIdx.x]*invSqrtP;}

    // k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
        float cosval = cosf(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
        tbr[threadIdx.x] = xn*sqrt2*cosval;
        if (needSync)   { __syncthreads();}
        reduceByN<float3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {xsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    // k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
        float  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
        tbr[threadIdx.x] = xn*sqrt2*sinval;
        if (needSync)   { __syncthreads();}
        reduceByN<float3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {xsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    // %%%%%%%%%%% VELOCITIES %%%%%%%%%%%
	// k = 0, n = 1,...,P
    tbr[threadIdx.x]  = vn;
    if (needSync)   { __syncthreads();}
    reduceByN<float3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {vsNM[threadIdx.x] = tbr[threadIdx.x]*invSqrtP;}

    // k = P/2, n = 1,...,P
    if (threadIdx.x % 2 == 0) {
        tbr[threadIdx.x] = vn * -1;
    } else {
        tbr[threadIdx.x] = vn ;
    }
    if (needSync)   { __syncthreads();}
    reduceByN<float3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {vsNM[threadIdx.x+halfP] = tbr[threadIdx.x]*invSqrtP;}

	// k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
        float cosval = cosf(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
        tbr[threadIdx.x] = vn*sqrt2*cosval;
        if (needSync)   { __syncthreads();}
        reduceByN<float3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {vsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

	// k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
	    float  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
        tbr[threadIdx.x] = vn*sqrt2*sinval;
        if (needSync)   { __syncthreads();}
        reduceByN<float3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {vsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    if (useThread ) {

	    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    // 2. NORMAL-MODE RP COORDINATE EVOLUTION
	    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // here each bead will handle the evolution of a particular normal-mode coordinate
	    // xk(t+dt) = xk(t)*cos(om_k*dt) + vk(t)*sinf(om_k*dt)/om_k
	    // vk(t+dt) = vk(t)*cosf(om_k*dt) - xk(t)*sinf(om_k*dt)*om_k
	    // k = 0
        if (amRoot) {
            xsNM[threadIdx.x] += vsNM[threadIdx.x] * dt; 
        } else {
	        float omegaK = 2.0f * omegaP * sinf( beadIdx * twoPiInvP * 0.5f);
	        float cosdt  = cosf(omegaK * dt);
	        float sindt  = sinf(omegaK * dt);
	        float3 xsNMk = xsNM[threadIdx.x];
	        float3 vsNMk = vsNM[threadIdx.x];
	        xsNM[threadIdx.x] *= cosdt;
	        vsNM[threadIdx.x] *= cosdt;
	        xsNM[threadIdx.x] += vsNMk * sindt / omegaK;
	        vsNM[threadIdx.x] -= xsNMk * sindt * omegaK;
        }
    }
    if (needSync) {__syncthreads();}

    if (useThread) {

	    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    // 3. COORDINATE BACK TRANSFORMATION
	    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    // k = 0
        xn = xsNM[rootIdx];
        vn = vsNM[rootIdx];
	    // k = halfP
	    // xn += xsNM[halfP]*(-1)**n
        if (threadIdx.x % 2) {
            xn += xsNM[rootIdx+halfP];
            vn += vsNM[rootIdx+halfP];
        } else {
            xn -= xsNM[rootIdx+halfP];
            vn -= vsNM[rootIdx+halfP];
        }

	    // k = 1,...,P/2-1; n = 1,...,P
        for (int k = 1; k < halfP; k++) {
	        float  cosval = cosf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
	        xn += xsNM[rootIdx+k] * sqrt2 * cosval;
	        vn += vsNM[rootIdx+k] * sqrt2 * cosval;
        }

	    // k = P/2+1,...,P-1; n = 1,...,P
        for (int k = halfP+1; k < nPerRingPoly; k++) {
	        float  sinval = sinf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
	        xn += xsNM[rootIdx+k] * sqrt2 * sinval;
	        vn += vsNM[rootIdx+k] * sqrt2 * sinval;
        }

	    // replace evolved back-transformation
        xn *= invSqrtP;
        vn *= invSqrtP;
	    xs[idx]   = make_float4(xn.x,xn.y,xn.z,xW);
	    vs[idx]   = make_float4(vn.x,vn.y,vn.z,vW);
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

        if (invmass > INVMASSBOOL ) {
            fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
            vs[idx] = make_float4(0.0f, 0.0f, 0.0f, invmass);
            return;
        }

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
// need to pass nPerRingPoly and omega_P
__global__ void preForcePIMD_cu(int nAtoms, int nPerRingPoly, float omegaP, float4 *xs, float4 *vs, float4 *fs,
                            float dt, float dtf)
{
    // Declare relevant variables for NM transformation
    int idx = GETIDX(); 
    extern __shared__ float3 xsvs[];
    float3 *xsNM = xsvs;				// normal-mode transform of position
    float3 *vsNM = xsvs + PERBLOCK;		// normal-mode transform of velocity
    float3 *tbr  = xsvs + 2*PERBLOCK;   // working array to place variables "to be reduced"
    bool useThread = idx < nAtoms;
    float3 xn = make_float3(0, 0, 0);
    float3 vn = make_float3(0, 0, 0);
    float xW;
    float vW;
    // helpful reference indices/identifiers
    bool   needSync= nPerRingPoly>warpSize;
    bool   amRoot  = (threadIdx.x % nPerRingPoly) == 0;
    int    rootIdx = (threadIdx.x / nPerRingPoly) * nPerRingPoly;
    int    beadIdx = idx % nPerRingPoly;
    int    n       = beadIdx + 1;

    // Update velocity by a half timestep for all beads in the ring polymer
    if (useThread) {
        float4 vel     = vs[idx];
        float  invmass = vel.w;
        float4 force   = fs[idx];
        // if its a massless particle, nothing more to be done here.
        if (invmass > INVMASSBOOL) {
            vs[idx] = make_float4(0.0, 0.0, 0.0, invmass);
            fs[idx] = make_float4(0.0, 0.0, 0.0, force.w);
            return;
        }
        float3 dv      = dtf * invmass * make_float3(force);
        vel           += dv;
        vs[idx]        = vel;
        fs[idx]        = make_float4(0.0f,0.0f,0.0f,force.w); // reset forces to zero before force calculation
    }
    //NOT SYNCED

    // 1. Transform to normal mode positions and velocities
    // 	xNM_k = \sum_{n=1}^P x_n* Cnk
    // 	Cnk = \sqrt(1/P)			k = 0
    // 	Cnk = \sqrt(2/P) cosf(2*pi*k*n/P)	1<= k <= P/2 -1
    // 	Cnk = \sqrt(1/P)(-1)^n			k = P/2
    // 	Cnk = \sqrt(2/P) sinf(2*pi*k*n/P)	P/2+1<= k <= P -1
    // 2. advance positions/velocities by full timestep according
    // to free ring-polymer evolution
    // 3. back transform to regular coordinates
    float invP            = 1.0f / (float) nPerRingPoly;
    float twoPiInvP       = 2.0f * M_PI * invP;
    float invSqrtP 	      = sqrtf(invP);
    float sqrt2           = sqrtf(2.0f);
    int   halfP           = nPerRingPoly / 2;	// P must be even for the following transformation!!!

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 1. COORDINATE TRANSFORMATION
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // first we will compute the normal mode coordinates for the positions, and then the velocities
    // using an identical structure. Each thread (bead) will contribute to each mode index, and
    // the strategy will be too store the results of each bead for a given mode in a working array
    // reduce the result by summation and store that in the final array for generating the positions

    // %%%%%%%%%%% POSITIONS %%%%%%%%%%%
    if (useThread) {
        float4 xWhole = xs[idx];
        float4 vWhole = vs[idx];
        xn = make_float3(xWhole);
        vn = make_float3(vWhole);
        xW = xWhole.w;
        vW = vWhole.w;
    }
    //STILL NOT SYNCED
    // k = 0, n = 1,...,P
    tbr[threadIdx.x]  = xn;
    if (needSync)    __syncthreads();
    reduceByN<float3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {xsNM[threadIdx.x] = tbr[threadIdx.x]*invSqrtP;}
    //SYNCED

    // k = P/2, n = 1,...,P
    if (threadIdx.x % 2 == 0) {
        tbr[threadIdx.x] = xn * -1;
    } else {
        tbr[threadIdx.x] = xn ;
    }
    if (needSync)   { __syncthreads();}
    reduceByN<float3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {xsNM[threadIdx.x+halfP] = tbr[threadIdx.x]*invSqrtP;}

    // k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
        float cosval = cosf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
        tbr[threadIdx.x] = xn*sqrt2*cosval;
        if (needSync)   { __syncthreads();}
        reduceByN<float3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {xsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    // k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
        float  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
        tbr[threadIdx.x] = xn*sqrt2*sinval;
        if (needSync)   { __syncthreads();}
        reduceByN<float3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {xsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    // %%%%%%%%%%% VELOCITIES %%%%%%%%%%%
	// k = 0, n = 1,...,P
    tbr[threadIdx.x]  = vn;
    if (needSync)   { __syncthreads();}
    reduceByN<float3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {vsNM[threadIdx.x] = tbr[threadIdx.x]*invSqrtP;}

    // k = P/2, n = 1,...,P
    if (threadIdx.x % 2 == 0) {
        tbr[threadIdx.x] = vn * -1;
    } else {
        tbr[threadIdx.x] = vn ;
    }
    if (needSync)   { __syncthreads();}
    reduceByN<float3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {vsNM[threadIdx.x+halfP] = tbr[threadIdx.x]*invSqrtP;}

	// k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
        float cosval = cosf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
        tbr[threadIdx.x] = vn*sqrt2*cosval;
        if (needSync)   { __syncthreads();}
        reduceByN<float3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {vsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

	// k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
	    float  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
        tbr[threadIdx.x] = vn*sqrt2*sinval;
        if (needSync)   { __syncthreads();}
        reduceByN<float3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {vsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    if (useThread ) {

	    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    // 2. NORMAL-MODE RP COORDINATE EVOLUTION
	    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // here each bead will handle the evolution of a particular normal-mode coordinate
	    // xk(t+dt) = xk(t)*cosf(om_k*dt) + vk(t)*sinf(om_k*dt)/om_k
	    // vk(t+dt) = vk(t)*cosf(om_k*dt) - xk(t)*sinf(om_k*dt)*om_k
	    // k = 0
        if (amRoot) {
            xsNM[threadIdx.x] += vsNM[threadIdx.x] * dt; 
        } else {
	        float omegaK = 2.0f * omegaP * sinf( beadIdx * twoPiInvP * 0.5);
	        float cosdt  = cosf(omegaK * dt);
	        float sindt  = sinf(omegaK * dt);
	        float3 xsNMk = xsNM[threadIdx.x];
	        float3 vsNMk = vsNM[threadIdx.x];
	        xsNM[threadIdx.x] *= cosdt;
	        vsNM[threadIdx.x] *= cosdt;
	        xsNM[threadIdx.x] += vsNMk * sindt / omegaK;
	        vsNM[threadIdx.x] -= xsNMk * sindt * omegaK;
        }
    }
    if (needSync) {__syncthreads();}

    if (useThread) {
	    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    // 3. COORDINATE BACK TRANSFORMATION
	    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    // k = 0
        xn = xsNM[rootIdx];
        vn = vsNM[rootIdx];
	    // k = halfP
	    // xn += xsNM[halfP]*(-1)**n
        if (threadIdx.x % 2) {
//POTENTIAL PROBLEM
            xn += xsNM[rootIdx+halfP];
            vn += vsNM[rootIdx+halfP];
        } else {//THIS TOO
            xn -= xsNM[rootIdx+halfP];
            vn -= vsNM[rootIdx+halfP];
        }

	    // k = 1,...,P/2-1; n = 1,...,P
        for (int k = 1; k < halfP; k++) {
	        float  cosval = cosf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
	        xn += xsNM[rootIdx+k] * sqrt2 * cosval;
	        vn += vsNM[rootIdx+k] * sqrt2 * cosval;
        }

	    // k = P/2+1,...,P-1; n = 1,...,P
        for (int k = halfP+1; k < nPerRingPoly; k++) {
	        float  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
	        xn += xsNM[rootIdx+k] * sqrt2 * sinval;
	        vn += vsNM[rootIdx+k] * sqrt2 * sinval;
        }

	    // replace evolved back-transformation
        xn *= invSqrtP;
        vn *= invSqrtP;
	    xs[idx]   = make_float4(xn.x,xn.y,xn.z,xW);
	    vs[idx]   = make_float4(vn.x,vn.y,vn.z,vW);
    }
}
    //if (useThread && amRoot ) {
    //    printf("--xx = %f\n",xs[idx].x);
    //    printf("--vx = %f\n",vs[idx].x);
    //    printf("--fx = %f\n",fs[idx].x);
    //    printf("R = np.array([");
    //    for (int i = 0; i <nPerRingPoly; i++) {
    //        printf("%f, ",xs[threadIdx.x+i].x);
    //    }
    //    printf("])\n");
    //    printf("V = np.array([");
    //    for (int i = 0; i <nPerRingPoly; i++) {
    //        printf("%f, ",vs[threadIdx.x+i].x);
    //    }
    //    printf("])\n");
    //}

__global__ void postForce_cu(int nAtoms, float4 *vs, float4 *fs, float dtf)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocities by a halftimestep
        float4 vel = vs[idx];
        float invmass = vel.w;
        if (invmass > INVMASSBOOL) {
            vs[idx] = make_float4(0.0f, 0.0f, 0.0f, invmass);
            return;
        }

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
    std::vector<bool> prepared = basicPrepare(numTurns); //nlist built here
    force(true);

    for (Fix *f : state->fixes) {
        if (!(f->prepared) ) {
            bool isPrepared = f->prepareForRun();
            if (!isPrepared) {
                mdError("A fix is unable to be instantiated correctly.");
            }
        }
    }

    /*
    for (int i = 0; i<prepared.size(); i++) {
        if (!prepared[i]) {
            for (Fix *f : state->fixes) {
                bool isPrepared = f->prepareForRun();
                if (!isPrepared) {
                    mdError("A fix is unable to be instantiated correctly.");
                }
            }
        }
    }
    */

    int periodicInterval = state->periodicInterval;

    // we should prepare for the datacomputers after the fixes
    prepareDataComputers();

    for (Fix *f : state->fixes) {
        f->assignLocalTempComputer();
    }

	
    auto start = std::chrono::high_resolution_clock::now();
    DataManager &dataManager = state->dataManager;
    dtf = 0.5f * state->dt * state->units.ftm_to_v;
    for (int i=0; i<numTurns; ++i) {
        if (state->turn % periodicInterval == 0) {
            state->gridGPU.periodicBoundaryConditions();
        }
        int virialMode = dataManager.getVirialModeForTurn(state->turn);

        stepInit(virialMode==1 or virialMode==2);

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
        force(virialMode);

        //quits if ctrl+c has been pressed
        checkQuit();

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
	    float omegaP    = (float) state->units.boltz * temp / state->units.hbar  ;
    	nve_xPIMD_cu<<<NBLOCK(state->atoms.size()), PERBLOCK, sizeof(float3) * 3 *PERBLOCK>>>(
	        state->atoms.size(),
	 	    nPerRingPoly,
		    omegaP,
    	    state->gpd.xs.getDevData(),
    	    state->gpd.vs.getDevData(),
    	    state->dt); 
    }
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
	    double temp;
	    for (Fix *f: state->fixes) {
	        if ( f->isThermostat && f->groupHandle == "all" ) {
	            std::string t = "temp";
	            temp = f->getInterpolator(t)->getCurrentVal();
	        }
	    }
        
	    int   nPerRingPoly = state->nPerRingPoly;
        int   nRingPoly    = state->atoms.size() / nPerRingPoly;
	    float omegaP       = (float) state->units.boltz * temp / state->units.hbar ;
   
        // called on a per bead basis
        preForcePIMD_cu<<<NBLOCK(state->atoms.size()), PERBLOCK, sizeof(float3) * 3 *PERBLOCK >>>(
	        state->atoms.size(),
	     	nPerRingPoly,
	    	omegaP,
        	state->gpd.xs.getDevData(),
        	state->gpd.vs.getDevData(),
        	state->gpd.fs.getDevData(),
        	state->dt,
        	dtf ); 
    }
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
    .def("run", &IntegratorVerlet::run,(py::arg("numTurns")))
    ;
}
