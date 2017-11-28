#pragma once

#include "cutils_math.h"

class EvaluatorCHARMM {
    public:
        inline __device__ real3 force(real3 dr, real params[5], real lenSqr, real multiplier) {
            if (multiplier) {
                bool isNorm = multiplier != mult14;
                real epstimes24 = isNorm ? params[1] : params[3];
                real sig6 = isNorm ? params[2] : params[4];
                real p1 = epstimes24*2*sig6*sig6;
                real p2 = epstimes24*sig6;
                real r2inv = 1.0/lenSqr;
                real r6inv = r2inv*r2inv*r2inv;
                real forceScalar = r6inv * r2inv * (p1 * r6inv - p2) * multiplier;
                return dr * forceScalar;
            }
            return make_real3(0, 0, 0);
        }



        inline __device__ real energy(real params[3], real lenSqr, real multiplier) {
            if (multiplier) {
                bool isNorm = multiplier != mult14;
                real eps = (isNorm ? params[1] : params[3]) / 24.0;
                real sig6 = isNorm ? params[2] : params[4];
                real r2inv = 1.0/lenSqr;
                real r6inv = r2inv*r2inv*r2inv;
                real sig6r6inv = sig6 * r6inv;
                real rCutSqr = params[0];
                real rCut6 = rCutSqr*rCutSqr*rCutSqr;

                real sig6InvRCut6 = sig6 / rCut6;
                real offsetOver4Eps = sig6InvRCut6*(sig6InvRCut6-1.0);
                return 0.5 * 4.0*eps*(sig6r6inv*(sig6r6inv-1.0) - offsetOver4Eps) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
            }
            return 0;
        }
        real mult14;
        EvaluatorCHARMM(real mult14_) {
            mult14 = mult14_;
        }

};

