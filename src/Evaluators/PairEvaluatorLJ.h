#pragma once
#ifndef EVALUATOR_LJ
#define EVALUATOR_LJ

#include "cutils_math.h"


void export_EvaluatorLJ();

class EvaluatorLJ {
    public:
        
        // default ctor 
        EvaluatorLJ();
        
        // host and device for testing
        inline __host__ __device__ real3 force(real3 dr, real params[3], real lenSqr, real multiplier) {
            if (multiplier) {
                real epstimes24 = params[1];
                real sig6 = params[2];
                real p1 = epstimes24*2*sig6*sig6;
                real p2 = epstimes24*sig6;
                real r2inv = 1.0/lenSqr;
                real r6inv = r2inv*r2inv*r2inv;
                real forceScalar = r6inv * r2inv * (p1 * r6inv - p2) * multiplier;
                return dr * forceScalar;
            }
            return make_real3(0, 0, 0);
        }
        

        // host and device for testing
        // full expression: 0.5 * (4 * epsilon ( ( sigma / r) ^12  - (sigma / r) ^6) + shift),
        // 0.5 from double counting, and shift is for the cutoff
        inline __host__ __device__ real energy(real params[3], real lenSqr, real multiplier) {
            if (multiplier) {
#ifdef DASH_DOUBLE
                real eps = params[1] / 24.0;
#else 
                real eps = params[1] / 24.0f;
#endif
                real sig6 = params[2];
                real r2inv = 1.0/lenSqr;
                real r6inv = r2inv*r2inv*r2inv;
                real sig6r6inv = sig6 * r6inv;
                real rCutSqr = params[0];
                real rCut6 = rCutSqr*rCutSqr*rCutSqr;

                real sig6InvRCut6 = sig6 / rCut6;
#ifdef DASH_DOUBLE
                real offsetOver4Eps = sig6InvRCut6*(sig6InvRCut6-1.0);
                return 0.5 * 4.0*eps*(sig6r6inv*(sig6r6inv-1.0) - offsetOver4Eps) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
#else
                real offsetOver4Eps = sig6InvRCut6*(sig6InvRCut6-1.0f);
                return 0.5f * 4.0f*eps*(sig6r6inv*(sig6r6inv-1.0f) - offsetOver4Eps) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant

#endif
            }
            return 0;
        }
        
        // python interface that calls force function
        __host__ Vector forcePy(double sigma, double epsilon, double rcut, Vector dr);

        // python interface that calls the energy function
        __host__ double energyPy(double sigma, double epsilon, double rcut, double distance);

        // python interface that calls the force function, and calculates on the GPU
        __host__ Vector forcePy_device(double sigma, double epsilon, double rcut, Vector dr);

        // python interface that calls the energy function, and calculates on the GPU
        __host__ double energyPy_device(double sigma, double epsilon, double rcut, double distance);
};

#endif
