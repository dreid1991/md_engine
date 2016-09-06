#pragma once
#ifndef EVALUATOR_DSF_
#define EVALUATOR_DSF

#include "cutils_math.h"

class ChargeEvaluatorDSF {
    public:
        float alpha;
        float A;
        float shift;
        inline __device__ float3 force(float3 dr, float lenSqr, float qi, float qj, float multiplier) {
            float r2inv = 1.0f/lenSqr;
            float rinv = sqrtf(r2inv);
            float len = sqrtf(lenSqr);
            float forceScalar = qi*qj*(erfcf((alpha*len))*r2inv+A*expf(-alpha*alpha*lenSqr)*rinv-shift)*rinv * multiplier;
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
        ChargeEvaluatorDSF(float alpha_, float A_, float shift_) : alpha(alpha_), A(A_), shift(shift_) {};

};

#endif
