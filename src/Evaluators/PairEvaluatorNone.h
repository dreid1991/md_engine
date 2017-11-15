#pragma once

#include "cutils_math.h"

class EvaluatorNone {
    public:
        char x; //variables on device must have non-zero size;
        inline __device__ real3 force(real3 dr, real params[1], real lenSqr, real multiplier) {
            return make_real3(0, 0, 0);
        }
        inline __device__ real energy(real params[0], real lenSqr, real multiplier) {
            return 0;
        }

};

