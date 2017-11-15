#pragma once
#ifndef BONDEVALUATORHARMONIC_H
#define BONDEVALUATORHARMONIC_H

#include "Bond.h"

class BondEvaluatorHarmonic {
public:
    inline __device__ real3 force(real3 bondVec, real rSqr, BondHarmonicType bondType) {
        real r = sqrtf(rSqr);
        real dr = r - bondType.r0;
        real rk = bondType.k * dr;
        if (r > 0) {//MAKE SURE ALL THIS WORKS, I JUST BORROWED FROM LAMMPS
            real fBond = -rk/r;
            return bondVec * fBond;
        } 
        return make_real3(0, 0, 0);
    }




    inline __device__ real energy(real3 bondVec, real rSqr, BondHarmonicType bondType) {
        real r = sqrtf(rSqr);
        real dr = r - bondType.r0;
        //printf("%f\n", (bondType.k/2.0) * 0.066 / (3.5*3.5));
        real eng = bondType.k * dr * dr * 0.5f;
        return 0.5f * eng; //0.5 for splitting between atoms
    }
};
#endif
