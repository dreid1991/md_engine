#pragma once
#ifndef BONDEVALUATORQUARTIC_H
#define BONDEVALUATORQUARTIC_H

#include "Bond.h"

class BondEvaluatorQuartic {
public:
    inline __device__ real3 force(real3 bondVec, real rSqr, BondQuarticType bondType) {
        real r = sqrtf(rSqr);
        if (r > 0) {
            real dr = r - bondType.r0;
            real dr2= dr*dr;
            real dr3= dr2*dr;
            real dUdr = 2.0*bondType.k2*dr + 3.0*bondType.k3*dr2 + 4.0*bondType.k4*dr3;
            real fBond = -dUdr/r;
            return bondVec * fBond;
        } 
        return make_real3(0, 0, 0);
    }



    inline __device__ real energy(real3 bondVec, real rSqr, BondQuarticType bondType) {
#ifdef DASH_DOUBLE
        real r = sqrt(rSqr);
#else
        real r = sqrtf(rSqr);
#endif
        real dr = r - bondType.r0;
        real dr2= dr*dr;
        real dr3= dr2*dr;
        real dr4= dr2*dr2;
        real eng = bondType.k2*dr2 + bondType.k3*dr3 + bondType.k4*dr4;
        return 0.5 * eng; //0.5 for splitting between atoms
    }
};
#endif
