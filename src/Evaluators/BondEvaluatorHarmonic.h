#pragma once
#ifndef BONDEVALUATORHARMONIC_H
#define BONDEVALUATORHARMONIC_H

#include "Bond.h"

class BondEvaluatorHarmonic {
public:
    inline __device__ float3 force(float3 bondVec, float r, BondHarmonicType bondType) {
        float dr = r - bondType.rEq;
        float rk = bondType.k * dr;
        if (r > 0) {//MAKE SURE ALL THIS WORKS, I JUST BORROWED FROM LAMMPS
            float fBond = -rk/r;
            return bondVec * fBond;
        } 
        return make_float3(0, 0, 0);


    }
};
#endif
