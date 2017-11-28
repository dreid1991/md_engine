#pragma once
#ifndef BONDEVALUATORFENE_H
#define BONDEVALUATORFENE_H

#include "Bond.h"

class BondEvaluatorFENE{
public:
    inline __device__ real3 force(real3 bondVec, real rSqr, BondFENEType bondType) {
        real k = bondType.k;
        real r0 = bondType.r0;
        real eps = bondType.eps;
        real sig = bondType.sig;
        real r0Sqr = r0*r0;
        real rlogarg = 1.0 - rSqr / r0Sqr;
        if (rlogarg < .1) {
            if (rlogarg < -3.0) {
                printf("FENE bond too long\n");
            }
            rlogarg = 0.1;
        }
        real fbond = -k / rlogarg;
#ifdef DASH_DOUBLE
        if (rSqr < pow(2.0, 1.0/3.0) * sig * sig) {
            real sr2 = sig*sig/rSqr;
            real sr6 = sr2*sr2*sr2;
            fbond += 48.0*eps*sr6*(sr6-0.5) / rSqr;
        }
#else
        if (rSqr < powf(2.0f, 1.0f/3.0f) * sig * sig) {
            real sr2 = sig*sig/rSqr;
            real sr6 = sr2*sr2*sr2;
            fbond += 48.0f*eps*sr6*(sr6-0.5f) / rSqr;

        }
#endif
        real3 force = bondVec * fbond;
        return force;
    }




    inline __device__ real energy(real3 bondVec, real rSqr, BondFENEType bondType) {
        real k = bondType.k;
        real r0 = bondType.r0;
        real eps = bondType.eps;
        real sig = bondType.sig;
        real sigOverR2 = sig*sig/rSqr;
#ifdef DASH_DOUBLE
        real sigOverR6 = pow(sigOverR2, 3.0);
        real eng = -0.5*k*r0*r0*log(1.0 - rSqr / (r0 * r0)) + 4.0*eps*(sigOverR6*sigOverR6 - sigOverR6) + eps;
#else
        real sigOverR6 = powf(sigOverR2, 3);
        real eng = -0.5f*k*r0*r0*log(1.0f - rSqr / (r0 * r0)) + 4.0*eps*(sigOverR6*sigOverR6 - sigOverR6) + eps;
#endif
        return 0.5 * eng; //0.5 for splitting between atoms
    }
};
#endif
