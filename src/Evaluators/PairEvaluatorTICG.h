#pragma once
#ifndef EVALUATOR_TICG
#define EVALUATOR_TICG

#include "cutils_math.h"

class EvaluatorTICG {
    public:
        inline __device__ void force(float3 &forceSum, float3 dr, float params[2], float lenSqr, float multiplier) {
            float rCutSqr = params[1];
            if (lenSqr < rCutSqr) {
              
              
                //float L = sqrt(lenSqr);
                //float rCut = sqrtf(rCutSqr);
                //here:
                //rSphere = 0.5*rCut
                //volume of normalized Spheres intersection is
                //V=1/(16*rSphere^3)*(L-2*rSphere)^2 *(L+4*rSphere);
                //or V=0.5/(rCut^3)*(L-rCut)^2 * (L+2*rcut);
                //then F=-dV/dL =-3/2 *(L^2-rCut^2)/rCut^3
                
              
                float forceScalar = (lenSqr!=0.0) ? -params[0]*1.5/sqrt(rCutSqr*lenSqr)*(lenSqr/rCutSqr-1.0)* multiplier: 0.0 ;

                float3 forceVec = dr * forceScalar;
                forceSum += forceVec;
            }
        }
        inline __device__ void energy(float &sumEng, float params[2], float lenSqr, float multiplier) {
            float rCutSqr = params[1];
            if (lenSqr < rCutSqr) {
              
                //float L = sqrt(lenSqr);
                //float rCut = sqrtf(rCutSqr);
                //here:
                //rSphere = 0.5*rCut
                //volume of Spheres intersection is
                //V=1/(16*rSphere^3)*(L-2*rSphere)^2 *(L+4*rSphere);
                //or V=0.5/(rCut^3)*(L-rCut)^2 * (L+2*rcut);   
                //E=1
                float Ldivrcut=sqrt(lenSqr/rCutSqr);
                float  V=0.5*(Ldivrcut-1.0)*(Ldivrcut-1.0)*(Ldivrcut+2.0);   
                sumEng += 0.5*params[0]*V * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
            }
        }

};

#endif
