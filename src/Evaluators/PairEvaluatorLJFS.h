#pragma once
#ifndef EVALUATOR_LJFS
#define EVALUATOR_LJFS
//Force-shifted Lennard-Jones Pair potential

#include "cutils_math.h"




class EvaluatorLJFS {
    public:
        inline __device__ real3 force(real3 dr, real params[4], real lenSqr, real multiplier) {
            if (multiplier) {
                real epstimes24 = params[1];
                real sig6 = params[2];
                real p1 = epstimes24*2.0*sig6*sig6;
                real p2 = epstimes24*sig6;
                real r2inv = 1.0/lenSqr;
                real r6inv = r2inv*r2inv*r2inv;
                real forceScalar = (r6inv * r2inv * (p1 * r6inv - p2)-params[3]/sqrt(lenSqr)) * multiplier ;

                return dr * forceScalar;
            }
            return make_real3(0, 0, 0);
        }
        
        inline __device__ real energy(real params[4], real lenSqr, real multiplier) {
            if (multiplier) {
                real epstimes24 = params[1];
                real sig6 = params[2];
                real r2inv = 1/lenSqr;
                real r6inv = r2inv*r2inv*r2inv;
                real sig6r6inv = sig6 * r6inv;
#ifdef DASH_DOUBLE
                return 0.5 * (4.0*(epstimes24 / 24.0)*sig6r6inv*(sig6r6inv-1.0)-params[3]*sqrt(lenSqr)) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
#else 
                return 0.5f * (4.0f*(epstimes24 / 24.0f)*sig6r6inv*(sig6r6inv-1.0f)-params[3]*sqrt(lenSqr)) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
#endif
            }
            return 0;
        }

};

#endif
