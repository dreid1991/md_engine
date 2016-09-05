#pragma once
#ifndef EVALUATOR_EWALD
#define EVALUATOR_EWALD

#include "cutils_math.h"

class EvaluatorEwald {
    public:
        float alpha;
        inline __device__ float3 force(float3 dr, float lenSqr, float qi, float qj, float multiplier) {
            float r2inv = 1.0f/lenSqr;
            float rinv = sqrtf(r2inv);
            float len = sqrtf(lenSqr);
            float forceScalar = qi*qj*(erfcf((alpha*len))*rinv+(2.0f*0.5641895835477563f*alpha)*exp(-alpha*alpha*lenSqr))*r2inv* multiplier;
            return dr * forceScalar;
        }
      /*  inline __device__ float energy(float params[3], float lenSqr, float multiplier) {
            float epstimes24 = params[1];
            float sig6 = params[2];
            float r2inv = 1/lenSqr;
            float r6inv = r2inv*r2inv*r2inv;
            float sig6r6inv = sig6 * r6inv;
            return 0.5f * 4*(epstimes24 / 24)*sig6r6inv*(sig6r6inv-1.0f) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
        }*/

};

#endif
