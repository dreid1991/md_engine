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
#include "FixTIP4PFlexible.h"

using namespace MD_ENGINE;
using std::cout;
using std::endl;

namespace py = boost::python;

__global__ void nve_v_cu(int nAtoms, real4 *vs, real4 *fs, real dtf) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        real4 vel = vs[idx];
        real invmass = vel.w;
        real4 force = fs[idx];
        
        // ghost particles should not have their velocities integrated; causes overflow
        if (invmass > INVMASSBOOL) {
            vs[idx] = make_real4(0.0, 0.0, 0.0,invmass);
            fs[idx] = make_real4(0.0, 0.0, 0.0,force.w);
            return;
        }

        //real3 dv = dtf * invmass * make_real3(force);
        real3 dv = dtf * invmass * make_real3(force);
        vel += dv;
        vs[idx] = vel;
        fs[idx] = make_real4(0.0, 0.0, 0.0, force.w);
    }
}

__global__ void nve_x_cu(int nAtoms, real4 *xs, real4 *vs, real dt) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update position by a full timestep
        real4 vel = vs[idx];
        real4 pos = xs[idx];

        //printf("pos %f %f %f\n", pos.x, pos.y, pos.z);
        //printf("vel %f %f %f\n", vel.x, vel.y, vel.z);
        //real3 dx = dt*make_real3(vel);
        real3 dx = dt*make_real3(vel);
        //printf("dx %f %f %f\n",dx.x, dx.y, dx.z);
        pos += dx;
        //xs[idx] = make_real4(pos);
        xs[idx] = pos;
    }
}

__global__ void nve_xPIMD_cu(int nAtoms, int nPerRingPoly, real omegaP, real4 *xs, real4 *vs, BoundsGPU bounds, real dt) {

    // Declare relevant variables for NM transformation
    int idx = GETIDX(); 
    extern __shared__ real3 xsvs[];
    real3 *xsNM = xsvs;				// normal-mode transform of position
    real3 *vsNM = xsvs + PERBLOCK;		// normal-mode transform of velocity
    real3 *tbr  = xsvs + 2*PERBLOCK;   // working array to place variables "to be reduced"
    bool useThread = idx < nAtoms;
    real3 xn = make_real3(0, 0, 0);
    real3 vn = make_real3(0, 0, 0);
    real xW;
    real vW;

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
#ifdef DASH_DOUBLE
	real invP            = 1.0 / (real) nPerRingPoly;
	real twoPiInvP       = 2.0 * M_PI * invP;
	real invSqrtP 	      = sqrt(invP);
	real sqrt2           = sqrt(2.0f);
#else
	real invP            = 1.0f / (real) nPerRingPoly;
	real twoPiInvP       = 2.0f * M_PI * invP;
	real invSqrtP 	      = sqrtf(invP);
	real sqrt2           = sqrtf(2.0f);
#endif /* DASH_DOUBLE */
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
        real4 xWhole = xs[idx];
        real4 vWhole = vs[idx];
        xn = make_real3(xWhole);
        vn = make_real3(vWhole);
        xW = xWhole.w;
        vW = vWhole.w;
    }
    // k = 0, n = 1,...,P
    tbr[threadIdx.x]  = xn;
    if (needSync)   __syncthreads();
    //taking care of PBC
    real3 origin = tbr[rootIdx];
    real3 deltaOrig = xn - origin;
    real3 deltaMin = bounds.minImage(deltaOrig);
    real3 wrapped = origin + deltaMin;
    xn = wrapped;
    tbr[threadIdx.x] = xn;
    

    if (needSync)    __syncthreads();
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (needSync)    __syncthreads();
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {xsNM[threadIdx.x] = tbr[threadIdx.x]*invSqrtP;}

    // k = P/2, n = 1,...,P
    if (threadIdx.x % 2 == 0) {
        tbr[threadIdx.x] = xn * -1;
    } else {
        tbr[threadIdx.x] = xn ;
    }
    if (needSync)   { __syncthreads();}
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {xsNM[threadIdx.x+halfP] = tbr[threadIdx.x]*invSqrtP;}

    // k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
#ifdef DASH_DOUBLE
        real cosval = cos(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
#else 
        real cosval = cosf(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
        tbr[threadIdx.x] = xn*sqrt2*cosval;
        if (needSync)   { __syncthreads();}
        reduceByN<real3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {xsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    // k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
#ifdef DASH_DOUBLE
        real  sinval = sin(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#else
        real  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
        tbr[threadIdx.x] = xn*sqrt2*sinval;
        if (needSync)   { __syncthreads();}
        reduceByN<real3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {xsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    // %%%%%%%%%%% VELOCITIES %%%%%%%%%%%
	// k = 0, n = 1,...,P
    tbr[threadIdx.x]  = vn;
    if (needSync)   { __syncthreads();}
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {vsNM[threadIdx.x] = tbr[threadIdx.x]*invSqrtP;}

    // k = P/2, n = 1,...,P
    if (threadIdx.x % 2 == 0) {
        tbr[threadIdx.x] = vn * -1;
    } else {
        tbr[threadIdx.x] = vn ;
    }
    if (needSync)   { __syncthreads();}
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {vsNM[threadIdx.x+halfP] = tbr[threadIdx.x]*invSqrtP;}

	// k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
#ifdef DASH_DOUBLE
        real cosval = cos(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
#else 
        real cosval = cosf(twoPiInvP * k * n);	// cos(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
        tbr[threadIdx.x] = vn*sqrt2*cosval;
        if (needSync)   { __syncthreads();}
        reduceByN<real3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {vsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

	// k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
#ifdef DASH_DOUBLE
	    real  sinval = sin(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#else
	    real  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
        tbr[threadIdx.x] = vn*sqrt2*sinval;
        if (needSync)   { __syncthreads();}
        reduceByN<real3>(tbr, nPerRingPoly, warpSize);
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
#ifdef DASH_DOUBLE
	        real omegaK = 2.0 * omegaP * sin( beadIdx * twoPiInvP * 0.5f);
	        real cosdt  = cos(omegaK * dt);
	        real sindt  = sin(omegaK * dt);
#else
	        real omegaK = 2.0f * omegaP * sinf( beadIdx * twoPiInvP * 0.5f);
	        real cosdt  = cosf(omegaK * dt);
	        real sindt  = sinf(omegaK * dt);
#endif /* DASH_DOUBLE */

	        real3 xsNMk = xsNM[threadIdx.x];
	        real3 vsNMk = vsNM[threadIdx.x];
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
#ifdef DASH_DOUBLE
	        real  cosval = cos(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#else 
            real  cosval = cosf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
	        xn += xsNM[rootIdx+k] * sqrt2 * cosval;
	        vn += vsNM[rootIdx+k] * sqrt2 * cosval;
        }

	    // k = P/2+1,...,P-1; n = 1,...,P
        for (int k = halfP+1; k < nPerRingPoly; k++) {
#ifdef DASH_DOUBLE
	        real  sinval = sin(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#else 
            real  sinval = sinf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
	        xn += xsNM[rootIdx+k] * sqrt2 * sinval;
	        vn += vsNM[rootIdx+k] * sqrt2 * sinval;
        }

	    // replace evolved back-transformation
        xn *= invSqrtP;
        vn *= invSqrtP;
	    xs[idx]   = make_real4(xn.x,xn.y,xn.z,xW);
	    vs[idx]   = make_real4(vn.x,vn.y,vn.z,vW);
    }

}

//so preForce_cu is split into two steps (nve_v, nve_x) if any of the fixes (barostat, for example), need to throw a step in there (as determined by requiresPostNVE_V flag)
__global__ void preForce_cu(int nAtoms, real4 *xs, real4 *vs, real4 *fs,
                            real dt, real dtf)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        real4 vel = vs[idx];
        real invmass = vel.w;
        real4 force = fs[idx];
        
        if (invmass > INVMASSBOOL) {
            vs[idx] = make_real4(0.0f, 0.0f, 0.0f,invmass);
            fs[idx] = make_real4(0.0f, 0.0f, 0.0f, force.w);
            return;
        }

        real3 dv = dtf * invmass * make_real3(force);
        vel += dv;
        vs[idx] = vel;

        // Update position by a full timestep
        real4 pos = xs[idx];

        //printf("vel %f %f %f\n", vel.x, vel.y, vel.z);
        real3 dx = dt*make_real3(vel);
        pos += dx;
        xs[idx] = pos;

        // Set forces to zero before force calculation
        fs[idx] = make_real4(0.0f, 0.0f, 0.0f, force.w);
    }
}

// alternative version of preForce_cu which allows for normal-mode propagation of RP dynamics
// need to pass nPerRingPoly and omega_P
__global__ void preForcePIMD_cu(int nAtoms, int nPerRingPoly, real omegaP, real4 *xs, real4 *vs, real4 *fs, BoundsGPU bounds,
                            real dt, real dtf)
{
    // Declare relevant variables for NM transformation
    int idx = GETIDX(); 
    extern __shared__ real3 xsvs[];
    real3 *xsNM = xsvs;				// normal-mode transform of position
    real3 *vsNM = xsvs + PERBLOCK;		// normal-mode transform of velocity
    real3 *tbr  = xsvs + 2*PERBLOCK;   // working array to place variables "to be reduced"
    bool useThread = idx < nAtoms;
    real3 xn = make_real3(0, 0, 0);
    real3 vn = make_real3(0, 0, 0);
    real xW;
    // helpful reference indices/identifiers
    bool   needSync= nPerRingPoly>warpSize;
    bool   amRoot  = (threadIdx.x % nPerRingPoly) == 0;
    int    rootIdx = (threadIdx.x / nPerRingPoly) * nPerRingPoly;
    int    beadIdx = idx % nPerRingPoly;
    int    n       = beadIdx + 1;

    // Update velocity by a half timestep for all beads in the ring polymer
    real4 vWhole;
    if (useThread) {
        vWhole = vs[idx];
        real4 force   = fs[idx];
        real3 dv      = dtf * vWhole.w * make_real3(force);
        vWhole        += dv;
        fs[idx]        = make_real4(0.0f,0.0f,0.0f,force.w); // reset forces to zero before force calculation
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
#ifdef DASH_DOUBLE
    real invP            = 1.0 / (real) nPerRingPoly;
    real twoPiInvP       = 2.0 * M_PI * invP;
    real invSqrtP 	      = sqrt(invP);
    real sqrt2           = sqrt(2.0);
#else
    real invP            = 1.0f / (real) nPerRingPoly;
    real twoPiInvP       = 2.0f * M_PI * invP;
    real invSqrtP 	      = sqrtf(invP);
    real sqrt2           = sqrtf(2.0f);
#endif /* DASH_DOUBLE */
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
        real4 xWhole = xs[idx];
        //real4 vWhole = vs[idx];
        xn = make_real3(xWhole);
        vn = make_real3(vWhole);
        xW = xWhole.w;
    }
    //STILL NOT SYNCED
    // k = 0, n = 1,...,P
    tbr[threadIdx.x]  = xn;
    if (needSync)   __syncthreads();
    //taking care of PBC
    real3 origin = tbr[rootIdx];
    real3 deltaOrig = xn - origin;
    real3 deltaMin = bounds.minImage(deltaOrig);
    real3 wrapped = origin + deltaMin;
    xn = wrapped;
    tbr[threadIdx.x] = xn;
    

    if (needSync)    __syncthreads();
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {xsNM[threadIdx.x] = tbr[threadIdx.x]*invSqrtP;}
    //SYNCED

    // k = P/2, n = 1,...,P
    if (threadIdx.x % 2 == 0) {
        tbr[threadIdx.x] = xn * -1;
    } else {
        tbr[threadIdx.x] = xn ;
    }
    if (needSync)   { __syncthreads();}
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {xsNM[threadIdx.x+halfP] = tbr[threadIdx.x]*invSqrtP;}

    // k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
#ifdef DASH_DOUBLE
        real cosval = cos(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#else 
        real cosval = cosf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
        tbr[threadIdx.x] = xn*sqrt2*cosval;
        if (needSync)   { __syncthreads();}
        reduceByN<real3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {xsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    // k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
#ifdef DASH_DOUBLE
        real  sinval = sin(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#else 
        real  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
        tbr[threadIdx.x] = xn*sqrt2*sinval;
        if (needSync)   { __syncthreads();}
        reduceByN<real3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {xsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

    // %%%%%%%%%%% VELOCITIES %%%%%%%%%%%
	// k = 0, n = 1,...,P
    tbr[threadIdx.x]  = vn;
    if (needSync)   { __syncthreads();}
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {vsNM[threadIdx.x] = tbr[threadIdx.x]*invSqrtP;}

    // k = P/2, n = 1,...,P
    if (threadIdx.x % 2 == 0) {
        tbr[threadIdx.x] = vn * -1;
    } else {
        tbr[threadIdx.x] = vn ;
    }
    if (needSync)   { __syncthreads();}
    reduceByN<real3>(tbr, nPerRingPoly, warpSize);
    if (useThread && amRoot)     {vsNM[threadIdx.x+halfP] = tbr[threadIdx.x]*invSqrtP;}

	// k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
#ifdef DASH_DOUBLE
        real cosval = cos(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#else 
        real cosval = cosf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
        tbr[threadIdx.x] = vn*sqrt2*cosval;
        if (needSync)   { __syncthreads();}
        reduceByN<real3>(tbr, nPerRingPoly, warpSize);
        if (useThread && amRoot)     {vsNM[threadIdx.x+k] = tbr[threadIdx.x]*invSqrtP;}
    }

	// k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
#ifdef DASH_DOUBLE
	    real  sinval = sin(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#else 
        real  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
        tbr[threadIdx.x] = vn*sqrt2*sinval;
        if (needSync)   { __syncthreads();}
        reduceByN<real3>(tbr, nPerRingPoly, warpSize);
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
#ifdef DASH_DOUBLE
	        real omegaK = 2.0 * omegaP * sin( beadIdx * twoPiInvP * 0.5);
	        real cosdt  = cos(omegaK * dt);
	        real sindt  = sin(omegaK * dt);
#else 
            real omegaK = 2.0f * omegaP * sinf( beadIdx * twoPiInvP * 0.5f);
	        real cosdt  = cosf(omegaK * dt);
	        real sindt  = sinf(omegaK * dt);
#endif /* DASH_DOUBLE */
	        real3 xsNMk = xsNM[threadIdx.x];
	        real3 vsNMk = vsNM[threadIdx.x];
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
#ifdef DASH_DOUBLE
	        real  cosval = cos(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#else 
            real  cosval = cosf(twoPiInvP * k * n);	// cosf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
	        xn += xsNM[rootIdx+k] * sqrt2 * cosval;
	        vn += vsNM[rootIdx+k] * sqrt2 * cosval;
        }

	    // k = P/2+1,...,P-1; n = 1,...,P
        for (int k = halfP+1; k < nPerRingPoly; k++) {
#ifdef DASH_DOUBLE
	        real  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#else 
            real  sinval = sinf(twoPiInvP * k * n);	// sinf(2*pi*k*n/P)
#endif /* DASH_DOUBLE */
	        xn += xsNM[rootIdx+k] * sqrt2 * sinval;
	        vn += vsNM[rootIdx+k] * sqrt2 * sinval;
        }

	    // replace evolved back-transformation
        xn *= invSqrtP;
        vn *= invSqrtP;
	    xs[idx]   = make_real4(xn.x,xn.y,xn.z,xW);
	    vs[idx]   = make_real4(vn.x,vn.y,vn.z,vWhole.w);
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

__global__ void postForce_cu(int nAtoms, real4 *vs, real4 *fs, real dtf)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocities by a halftimestep
        real4 vel = vs[idx];
        real invmass = vel.w;
        if (invmass > INVMASSBOOL) {
            vs[idx] = make_real4(0.0f, 0.0f, 0.0f,invmass);
            return;
        }
        real4 force = fs[idx];

        real3 dv = dtf * invmass * make_real3(force);
        vel += dv;
        vs[idx] = vel;
    }
}

IntegratorVerlet::IntegratorVerlet(State *state_)
    : Integrator(state_)
{

}

void IntegratorVerlet::setInterpolator() {
    for (Fix *f: state->fixes) {
        if ( f->isThermostat && f->groupHandle == "all" ) {
            std::string t = "temp";
            tempInterpolator = f->getInterpolator(t);
            return;
        }
    }
    mdError("No thermostat found when setting up PIMD");
}


double IntegratorVerlet::run(int numTurns)
{

    basicPreRunChecks();
    
    // basicPrepare now only handles State prepare and sending global State data to device
    basicPrepare(numTurns);
    
    // prepare the fixes that do not require forces to be computed
    // -- e.g., isotropic pair potentials
    prepareFixes(false);
    
    // iterates over fixes & data computers, if any 'requireVirials' then true, else false
    bool initialVirialMode = getInitialVirialMode();
    
    // iterates and computes forces only from fixes that return (prepared==true); 
    // computes virials, if these are found to be needed
    forceInitial(initialVirialMode);
    
    // prepare the fixes that require forces to be computed on instantiation;
    // -- e.g., constraints
    prepareFixes(true);
    
    // finally, prepare barostats, thermostats, datacomputers, etc.
    // datacomputers are prepared first, then the barostats, thermostats, etc.
    // prior to datacomputers being prepared, we iterate over State, and the groups in simulation 
    // collect their NDF associated with their group
    prepareFinal();
   
    // get our PIMD thermostat
    if (state->nPerRingPoly>1) {
        setInterpolator();
    }

    std::cout << "about to enter the run loop.." << std::endl;
    verifyPrepared();
    
    int periodicInterval = state->periodicInterval;
	
    auto start = std::chrono::high_resolution_clock::now();

    DataManager &dataManager = state->dataManager;
    dtf = 0.5 * state->dt * state->units.ftm_to_v;
    int tuneEvery = state->tuneEvery;
    bool haveTunedWithData = false;
    double timeTune = 0;
    for (int i=0; i<numTurns; ++i) {

        if (state->turn % periodicInterval == 0 or state->turn == state->nextForceBuild) {
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

        handleLocalData();

        if (state->tuning) {
            if ((state->turn-state->runInit) % tuneEvery == 0 and state->turn > state->runInit) {
                //this goes here because forces are zero at this point.  I don't need to save any forces this way
                timeTune += tune();
            } else if (not haveTunedWithData and state->turn-state->runInit < tuneEvery and state->nlistBuildCount > 20) {
                timeTune += tune();
                haveTunedWithData = true;
            }
        }
        // Recalculate forces
        force(virialMode);

        //quits if ctrl+c has been pressed
        checkQuit();

        // Perform second half of velocity-Verlet step
        postForce();

        // for NPT rigid body simulations - add in the virial correction before a barostat computes the pressure
        // i.e. we do the velocity correction in this step
        preStepFinal();

        stepFinal();

        //HEY - MAKE DATA APPENDING HAPPEN WHILE SOMETHING IS GOING ON THE GPU.  
        doDataComputation();
        doDataAppending();
        dataManager.clearVirialTurn(state->turn);
        asyncOperations();

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
    double ptsps = state->atoms.size()*numTurns / (duration.count() - timeTune);
    mdMessage("runtime %f\n%e particle timesteps per second\n",
              duration.count(), ptsps);

    basicFinish();
    return ptsps;
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
	    double temp = tempInterpolator->getCurrentVal();
	    int   nPerRingPoly = state->nPerRingPoly;
        int   nRingPoly = state->atoms.size() / nPerRingPoly;
	    real omegaP    = (real) state->units.boltz * temp / state->units.hbar  ;
    	nve_xPIMD_cu<<<NBLOCK(state->atoms.size()), PERBLOCK, sizeof(real3) * 3 *PERBLOCK>>>(
	        state->atoms.size(),
	 	    nPerRingPoly,
		    omegaP,
    	    state->gpd.xs.getDevData(),
    	    state->gpd.vs.getDevData(),
            state->boundsGPU,
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
	    double temp = tempInterpolator->getCurrentVal();
        
	    int   nPerRingPoly = state->nPerRingPoly;
        int   nRingPoly    = state->atoms.size() / nPerRingPoly;
	    real omegaP       = (real) state->units.boltz * temp / state->units.hbar ;
   
        // called on a per bead basis
        preForcePIMD_cu<<<NBLOCK(state->atoms.size()), PERBLOCK, sizeof(real3) * 3 *PERBLOCK >>>(
	        state->atoms.size(),
	     	nPerRingPoly,
	    	omegaP,
        	state->gpd.xs.getDevData(),
        	state->gpd.vs.getDevData(),
        	state->gpd.fs.getDevData(),
            state->boundsGPU,
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
    /* FOR TESTING PURPOSES ONLY */
    /* These do not go in the user documentation, and should not be used in a production simulation */
    /* Methods inherited from IntegratorUtil, Integrator; and methods usually not exported to python */
    .def("preForce", &IntegratorVerlet::preForce)                   // IntegratorVerlet
    .def("postForce",&IntegratorVerlet::postForce)                  // IntegratorVerlet
    .def("nve_v",    &IntegratorVerlet::nve_v)                      // IntegratorVerlet
    .def("nve_x",    &IntegratorVerlet::nve_x)                      // IntegratorVerlet
    .def("stepInit", &IntegratorVerlet::stepInit,
         boost::python::arg("computeVirials")
         )                                                          // Integrator
    .def("stepFinal",&IntegratorVerlet::stepFinal)                  // Integrator
    .def("asyncOperations", &IntegratorVerlet::asyncOperations)     // Integrator
    .def("basicPreRunChecks", &IntegratorVerlet::basicPreRunChecks) // Integrator
    .def("basicPrepare", &IntegratorVerlet::basicPrepare,
         boost::python::arg("numTurns")
         )                                                          // Integrator
    .def("prepareFixes", &IntegratorVerlet::prepareFixes,
         boost::python::arg("requiresForces")
        )                                                           // Integrator
    .def("prepareFinal", &IntegratorVerlet::prepareFinal)           // Integrator
    .def("basicFinish",  &IntegratorVerlet::basicFinish)            // Integrator 
    .def("setActiveData", &IntegratorVerlet::setActiveData)         // Integrator
    .def("tune", &IntegratorVerlet::tune)                           // Integrator
    .def("verifyPrepared", &IntegratorVerlet::verifyPrepared)       // Integrator
    .def("force", &IntegratorVerlet::force,
         boost::python::arg("virialMode")
        )                                                            // IntegratorUtil
    .def("forceInitial", &IntegratorVerlet::forceInitial,
         boost::python::arg("virialMode")
        )                                                             // IntegratorUtil
    .def("postNVE_V", &IntegratorVerlet::postNVE_V)                   // IntegratorUtil
    .def("postNVE_X", &IntegratorVerlet::postNVE_X)                   // IntegratorUtil
    .def("handleBoundsChange", &IntegratorVerlet::handleBoundsChange) // IntegratorUtil
    ;
}
