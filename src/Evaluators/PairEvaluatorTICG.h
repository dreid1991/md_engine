#pragma once
#ifndef EVALUATOR_TICG
#define EVALUATOR_TICG

#include "cutils_math.h"

class EvaluatorTICG {
public:
    inline __device__ real3 force(real3 dr, real params[2], real lenSqr, real multiplier) {
        if (multiplier) {
            real rCutSqr = params[0];


            //real L = sqrt(lenSqr);
            //real rCut = sqrtf(rCutSqr);
            //here:
            //rSphere = 0.5*rCut
            //volume of normalized Spheres intersection is
            //V=1/(16*rSphere^3)*(L-2*rSphere)^2 *(L+4*rSphere);
            //or V=0.5/(rCut^3)*(L-rCut)^2 * (L+2*rcut);
            //then F=-dV/dL =-3/2 *(L^2-rCut^2)/rCut^3


            real forceScalar = (lenSqr!=0.0) ? -params[1]*1.5/sqrt(rCutSqr*lenSqr)*(lenSqr/rCutSqr-1.0)* multiplier: 0.0 ;

            return dr * forceScalar;
        }
        return make_real3(0, 0, 0);
    }


    inline __device__ real energy(real params[2], real lenSqr, real multiplier) {
        if (multiplier) {
            real rCutSqr = params[0];

            //real L = sqrt(lenSqr);
            //real rCut = sqrtf(rCutSqr);
            //here:
            //rSphere = 0.5*rCut
            //volume of Spheres intersection is
            //V=1/(16*rSphere^3)*(L-2*rSphere)^2 *(L+4*rSphere);
            //or V=0.5/(rCut^3)*(L-rCut)^2 * (L+2*rcut);   
            //E=1
            real Ldivrcut=sqrt(lenSqr/rCutSqr);
            real  V=0.5*(Ldivrcut-1.0)*(Ldivrcut-1.0)*(Ldivrcut+2.0);   
            return 0.5f*params[1]*V * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
        }
        return 0;
    }

};

#endif
