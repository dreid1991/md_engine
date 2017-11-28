#pragma once
#ifndef EVALUATOR_WCA
#define EVALUATOR_WCA
//Weeks Chandler Andersen potential (WCA)

#include "cutils_math.h"




class EvaluatorWCA {
public:
    inline __device__ real3 force(real3 dr, real params[3], real lenSqr, real multiplier) {
        if (multiplier) {
            real epstimes24 = params[1];
            real sig6 = params[2];
            real p1 = epstimes24*2.0*sig6*sig6;
            real p2 = epstimes24*sig6;
            real r2inv = 1.0/lenSqr;
            real r6inv = r2inv*r2inv*r2inv;
            real forceScalar = r6inv * r2inv * (p1 * r6inv - p2) * multiplier;

            return dr * forceScalar;
        }
        return make_real3(0, 0, 0);
    }
    
    inline __device__ real energy(real params[3], real lenSqr, real multiplier) {
        if (multiplier) {
            real eps = params[1]/24.0;
            real sig6 = params[2];
            real r2inv = 1.0/lenSqr;
            real r6inv = r2inv*r2inv*r2inv;
            real sig6r6inv = sig6 * r6inv;
            return 0.5 * (4.0*(eps)*sig6r6inv*(sig6r6inv-1.0)+eps) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
        }
        return 0;
    }

};

#endif
