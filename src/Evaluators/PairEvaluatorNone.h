#pragma once

#include "cutils_math.h"

class EvaluatorLJ {
    public:
        inline __device__ float3 force(float3 dr, float params[0], float lenSqr, float multiplier) {
            return make_float3(0, 0, 0);
        }
        inline __device__ float energy(float params[0], float lenSqr, float multiplier) {
            return 0;
        }

};

