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
    inline __device__ float energy(float3 bondVec, float r, BondHarmonicType bondType) {
        float dr = r - bondType.rEq;
        float eng = bondType.k * dr * dr * 0.5f;
        return 0.5f * eng; //0.5 for splitting between atoms
    }
};
#endif
