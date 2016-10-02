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
            float prefactor = qqr_to_eng * qi*qj*(erfcf((alpha*len))*rinv+(2.0f*0.5641895835477563f*alpha)*exp(-alpha*alpha*lenSqr));
            float forceScalar = prefactor * multiplier;
            if (multiplier < 1.0f) {
                forceScalar -= (1.0f - multiplier) * prefactor;
            }

            forceScalar *= r2inv;
            printf("EVALUATOR force scalar in eval is %f\n", forceScalar);
            return dr * forceScalar;
        }
        inline __device__ float energy(float lenSqr, float qi, float qj, float multiplier) {
             float len=sqrtf(lenSqr);
             float rinv = 1.0f/len;                 
             float prefactor = qqr_to_eng * qi*qj*(erfcf((alpha*len))*rinv);
             float eng = multiplier * prefactor;
             if (multiplier < 1.0f) {
                 eng -= 2.0f * (1.0f - multiplier) * prefactor;//gets multiplied by 0.5 further down, and need total coef on this to be 1
             }
             printf("ENG EVAL prefactor %f, mult %f, final %f\n", prefactor, multiplier, eng);
             return 0.5f * eng;
                   
        }
        ChargeEvaluatorEwald(float alpha_, float qqr_to_eng_) : alpha(alpha_), qqr_to_eng(qqr_to_eng_) {};

};

#endif
