#pragma once
#ifndef EVALUATOR_EWALD
#define EVALUATOR_EWALD

#include "cutils_math.h"

class ChargeEvaluatorEwald {
    public:
        real alpha;
        real qqr_to_eng;
        inline __device__ real3 force(real3 dr, real lenSqr, real qi, real qj, real multiplier) {
            if (lenSqr < 1e-10) {
                lenSqr = 1e-10;
            }
            real r2inv = 1.0f/lenSqr;
            real rinv = sqrtf(r2inv);
            real len = sqrtf(lenSqr);
            real forceScalar = qqr_to_eng * qi*qj*(erfcf((alpha*len))*rinv+(2.0f*0.5641895835477563f*alpha)*exp(-alpha*alpha*lenSqr));
            if (multiplier < 1.0f) {
                real correctionVal = qqr_to_eng * qi * qj * rinv;
                forceScalar -= (1.0f - multiplier) * correctionVal;
            }

            forceScalar *= r2inv;
            return dr * forceScalar;
        }


        inline __device__ real energy(real lenSqr, real qi, real qj, real multiplier) {
            if (lenSqr < 1e-10) {
                lenSqr = 1e-10;
            }
            real len=sqrtf(lenSqr);
            real rinv = 1.0f/len;                 
            real prefactor = qqr_to_eng * qi * qj * rinv;
            real eng = prefactor * erfcf(alpha*len);
            if (multiplier < 1.0f) {
                eng -= (1 - multiplier) * prefactor;
            }
            return 0.5f * eng;

        }
        ChargeEvaluatorEwald(real alpha_, real qqr_to_eng_) : alpha(alpha_), qqr_to_eng(qqr_to_eng_) {};

};

#endif
