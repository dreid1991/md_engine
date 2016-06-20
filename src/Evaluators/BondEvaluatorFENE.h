#pragma once
#ifndef BONDEVALUATORFENE_H
#define BONDEVALUATORFENE_H

#include "Bond.h"

class BondEvaluatorFENE{
public:
    inline __device__ float3 force(float3 bondVec, float r, BondFENEType bondType) {
        float k = bondType.k;
        float rEq = bondType.rEq;
        float eps = bondType.eps;
        float sig = bondType.sig;
        float rEqSqr = rEq*rEq;
        float rSqr = r*r; 
        float rlogarg = 1.0f - rSqr / rEqSqr;
        printf("%f %f %f %f %f\n", k, rEq, eps, sig, r);
        if (rlogarg < .1f) {
            if (rlogarg < -3.0f) {
                printf("FENE bond too long\n");
            }
            rlogarg = 0.1f;
        }
        float fbond = -k / rlogarg;
        if (rSqr < powf(2.0f, 1.0f/3.0f) * sig * sig) {
            float sr2 = sig*sig/rSqr;
            float sr6 = sr2*sr2*sr2;
            fbond += 48.0f*eps*sr6*(sr6-0.5f) / rSqr;
            return bondVec * fbond;

        }
        return make_float3(0, 0, 0);
    }
    inline __device__ float energy(float3 bondVec, float r, BondFENEType bondType) {
        float k = bondType.k;
        float rEq = bondType.rEq;
        float eps = bondType.eps;
        float sig = bondType.sig;
        float rOverReq = r / rEq;
        float sigOverR6 = powf(sig/r, 6);
        float eng = -0.5f*k*rEq*rEq*logf(1.0f - rOverReq*rOverReq) + 4*eps*(sigOverR6*sigOverR6 - sigOverR6) + eps;
        return 0.5f * eng; //0.5 for splitting between atoms
    }
};
#endif
