#pragma once

#include "cutils_math.h"

class EvaluatorEField {
    public:
        real angToBohr;
        //not a force - just vector representing the electric field
        inline __device__ real3 force(real3 dr, real lenSqr, real qi, real qj, real multiplier) {
            real3 inBohr = dr * angToBohr;
            real3 field = qj * inBohr / powf(lenSqr, 1.5);
            return field;
        }
        inline __device__ real energy(real lenSqr, real qi, real qj, real multiplier) {
            return 0;
        }
        EvaluatorEField() : angToBohr(1.88977161646) {};

};

