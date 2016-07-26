#pragma once
#ifndef IMPROPER_HARMONIC_H
#include "Improper.h"
class ImproperEvaluatorHarmonic {
public:
    inline __device__ float3 force(ImproperHarmonicType improperType, float theta, float scValues[3], float invLenSqrs[3], float invLens[3], float angleBits[3], float s, float c, float3 directors[3], int myIdxInImproper) {
        float dTheta = theta - improperType.thetaEq;

        float a = improperType.k * dTheta;
        a *= -2.0f / s;
        scValues[2] *= a;
        c *= a;
        float a11 = c * invLenSqrs[0] * scValues[0];
        float a22 = - invLenSqrs[1] * (2.0f * angleBits[0] * scValues[2] - c * (scValues[0] + scValues[1]));
        float a33 = c * invLenSqrs[2] * scValues[1];
        float a12 = -invLens[0] * invLens[1] * (angleBits[1] * c * scValues[0] + angleBits[2] * scValues[2]);
        float a13 = -invLens[0] * invLens[2] * scValues[2];
        float a23 = invLens[1] * invLens[2] * (angleBits[2] * c * scValues[1] + angleBits[1] * scValues[2]);

        float3 myForce = make_float3(0, 0, 0);
        float3 sFloat3 = make_float3(
                                     a22*directors[1].x + a23*directors[2].x + a12*directors[0].x
                                     ,  a22*directors[1].y + a23*directors[2].y + a12*directors[0].y
                                     ,  a22*directors[1].z + a23*directors[2].z + a12*directors[0].z
                                    );
        if (myIdxInImproper <= 1) {
            float3 a11Dir1 = directors[0] * a11;
            float3 a12Dir2 = directors[1] * a12;
            float3 a13Dir3 = directors[2] * a13;
            myForce.x += a11Dir1.x + a12Dir2.x + a13Dir3.x;
            myForce.y += a11Dir1.y + a12Dir2.y + a13Dir3.y;
            myForce.z += a11Dir1.z + a12Dir2.z + a13Dir3.z;

            if (myIdxInImproper == 1) {
                myForce = -sFloat3 - myForce;
            }
        } else {
            float3 a13Dir1 = directors[0] * a13;
            float3 a23Dir2 = directors[1] * a23;
            float3 a33Dir3 = directors[2] * a33;
            myForce.x += a13Dir1.x + a23Dir2.x + a33Dir3.x;
            myForce.y += a13Dir1.y + a23Dir2.y + a33Dir3.y;
            myForce.z += a13Dir1.z + a23Dir2.z + a33Dir3.z;
            if (myIdxInImproper == 2) {
                myForce = sFloat3 - myForce;
            }
        }
        return myForce;
    }
   inline __device__ void forcesAll(ImproperHarmonicType improperType, float theta, float scValues[3], float invLenSqrs[3], float invLens[3], float angleBits[3], float s, float c, float3 directors[3], float3 forces[4]) {
        float dTheta = theta - improperType.thetaEq;

        float a = improperType.k * dTheta;
        a *= -2.0f / s;
        scValues[2] *= a;
        c *= a;
        float a11 = c * invLenSqrs[0] * scValues[0];
        float a22 = - invLenSqrs[1] * (2.0f * angleBits[0] * scValues[2] - c * (scValues[0] + scValues[1]));
        float a33 = c * invLenSqrs[2] * scValues[1];
        float a12 = -invLens[0] * invLens[1] * (angleBits[1] * c * scValues[0] + angleBits[2] * scValues[2]);
        float a13 = -invLens[0] * invLens[2] * scValues[2];
        float a23 = invLens[1] * invLens[2] * (angleBits[2] * c * scValues[1] + angleBits[1] * scValues[2]);

        float3 sFloat3 = make_float3(
                                     a22*directors[1].x + a23*directors[2].x + a12*directors[0].x
                                     ,  a22*directors[1].y + a23*directors[2].y + a12*directors[0].y
                                     ,  a22*directors[1].z + a23*directors[2].z + a12*directors[0].z
                                    );
        float3 a11Dir1 = directors[0] * a11;
        float3 a12Dir2 = directors[1] * a12;
        float3 a13Dir3 = directors[2] * a13;
        forces[0].x += a11Dir1.x + a12Dir2.x + a13Dir3.x;
        forces[0].y += a11Dir1.y + a12Dir2.y + a13Dir3.y;
        forces[0].z += a11Dir1.z + a12Dir2.z + a13Dir3.z;

        forces[1] = -sFloat3 - forces[0];
        float3 a13Dir1 = directors[0] * a13;
        float3 a23Dir2 = directors[1] * a23;
        float3 a33Dir3 = directors[2] * a33;
        forces[3].x += a13Dir1.x + a23Dir2.x + a33Dir3.x;
        forces[3].y += a13Dir1.y + a23Dir2.y + a33Dir3.y;
        forces[3].z += a13Dir1.z + a23Dir2.z + a33Dir3.z;
        forces[2] = sFloat3 - forces[3];
   }

    inline __device__ float energy(ImproperHarmonicType improperType, float theta, float scValues[3], float invLenSqrs[3], float invLens[3], float angleBits[3], float s, float c, float3 directors[3], int myIdxInImproper) {
        float dTheta = theta - improperType.thetaEq;
        return (1.0f/2.0f) * (1.0f/4.0f) * dTheta * dTheta * improperType.k;

    }


};

#endif

