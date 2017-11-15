#pragma once

#include "cutils_math.h"

class EvaluatorDipolarCoupling {
    public:
        //all the math with couplings, etc will be done on the CPU in double precision
        inline __device__ real3 force(real3 dr, real params[1], real lenSqr, real multiplier) {
            assert(0);
            return make_real3(0, 0, 0);
        }
        inline __device__ real energy(real params[1], real lenSqr, real multiplier) {
            //need to compute 1/r^6 because we want D^2
            return 1.0f / powf(lenSqr, 3);
        }

};

