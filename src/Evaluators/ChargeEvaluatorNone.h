#pragma once
#ifndef EVALUATOR_NONE
#define EVALUATOR_NONE

#include "cutils_math.h"

class ChargeEvaluatorNone {
    public:
        inline __device__ real3 force(real3 dr, real lenSqr, real qi, real qj, real multiplier) {
            return make_real3(0, 0, 0);
        }



        inline __device__ real energy(real lenSqr, real qi, real qj, real multiplier) {
            return 0;
        }
      /*  inline __device__ real energy(real params[3], real lenSqr, real multiplier) {
            real epstimes24 = params[1];
            real sig6 = params[2];
            real r2inv = 1/lenSqr;
            real r6inv = r2inv*r2inv*r2inv;
            real sig6r6inv = sig6 * r6inv;
            return 0.5f * 4*(epstimes24 / 24)*sig6r6inv*(sig6r6inv-1.0f) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
        }*/

};

#endif
