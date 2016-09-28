#pragma once
#ifndef EVALUATOR_EWALD
#define EVALUATOR_EWALD

#include "cutils_math.h"

class ChargeEvaluatorEwald {
    public:
        float alpha;
        float qqr_to_eng;
        inline __device__ float3 force(float3 dr, float lenSqr, float qi, float qj, float multiplier) {
            float r2inv = 1.0f/lenSqr;
            float rinv = sqrtf(r2inv);
            float len = sqrtf(lenSqr);
            float forceScalar = qqr_to_eng * qi*qj*(erfcf((alpha*len))*rinv+(2.0f*0.5641895835477563f*alpha)*exp(-alpha*alpha*lenSqr))*r2inv* multiplier;
            return dr * forceScalar;
        }
        inline __device__ float energy(float lenSqr, float qi, float qj, float multiplier) {
             float len=sqrtf(lenSqr);
             float rinv = 1.0f/len;                 
             float eng = qqr_to_eng * 0.5*qi*qj*(erfcf((alpha*len))*rinv)*multiplier;
             //printf("alpha %f len %f product %f over r %f\n", alpha, len, alpha*len, erfcf((alpha*len))*rinv);
             return eng;
                   
        }
        ChargeEvaluatorEwald(float alpha_, float qqr_to_eng_) : alpha(alpha_), qqr_to_eng(qqr_to_eng_) {};

};

#endif
