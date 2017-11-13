#pragma once

#include "cutils_math.h"

class ChargeEvaluatorDSF {
    public:
        real alpha;
        real A;
        real shift;
        real qqr_to_eng;
        real r_cut;
        inline __device__ real3 force(real3 dr, real lenSqr, real qi, real qj, real multiplier) {
            real r2inv = 1.0f/lenSqr;
            real rinv = sqrtf(r2inv);
            real len = sqrtf(lenSqr);
            real forceScalar = qqr_to_eng * qi*qj*(erfcf((alpha*len))*r2inv+A*expf(-alpha*alpha*lenSqr)*rinv-shift)*rinv * multiplier;
            return dr * forceScalar;
        }




        inline __device__ real energy(real lenSqr, real qi, real qj, real multiplier) {
            real r2inv = 1.0f/lenSqr;
            real rinv = sqrtf(r2inv);
            real len = sqrtf(lenSqr);
            real eng = qqr_to_eng * qi*qj*(erfcf(alpha*len)*rinv-erfcf(alpha*r_cut)/r_cut+shift*(len-r_cut))* multiplier;
            return eng*0.5;
        }
        ChargeEvaluatorDSF(real alpha_, real A_, real shift_, real qqr_to_eng_,real r_cut_) : alpha(alpha_), A(A_), shift(shift_), qqr_to_eng(qqr_to_eng_),r_cut(r_cut_) {};

};

