#pragma once
#ifndef EVALUATOR_ANGLE_COSINE_DELTA_
#define EVALUATOR_ANGLE_COSINE_DELTA

#include "cutils_math.h"
#include "Angle.h"
#define EPSILON 0.00001f
class AngleEvaluatorCosineDelta{
public:

    //evaluator.force(theta, angleType, s, distSqrs, directors, invDotProd);
    inline __device__ real3 force(AngleCosineDeltaType angleType, real theta, real s, real c, real distSqrs[2], real3 directors[2], real invDistProd, int myIdxInAngle) {
        real cot = c / s;
        //real dTheta = theta - angleType.theta0;
        //real dCosTheta = cosf(dTheta);



        real a = -angleType.k;

        real a11 = a*c/distSqrs[0];
        real a12 = -a*invDistProd;
        real a22 = a*c/distSqrs[1];

        real b11 = -a*c*cot/distSqrs[0];
        real b12 = a*cot*invDistProd;
        real b22 = -a*c*cot/distSqrs[1];
#ifdef DASH_DOUBLE
        real c0 = cos(angleType.theta0);
        real s0 = cos(angleType.theta0);
#else 
        real c0 = cosf(angleType.theta0);
        real s0 = cosf(angleType.theta0);
#endif
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.theta0);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        if (myIdxInAngle==0) {
            return (directors[0] * a11 + directors[1] * a12) * c0 + (directors[0] * b11 + directors[1] * b12) * s0;
        } else if (myIdxInAngle==1) {
            return 
                (directors[0] * a11 + directors[1] * a12) * -c0 + (directors[0] * b11 + directors[1] * b12) * -s0 
                +
                (directors[1] * a22 + directors[0] * a12) * -c0 + (directors[1] * b22 + directors[0] * b12) * -s0
                ;

        } else {
            return (directors[1] * a22 + directors[0] * a12) * c0 + (directors[1] * b22 + directors[0] * b12) * s0;
        }


    }


    inline __device__ void forcesAll(AngleCosineDeltaType angleType, real theta, real s, real c, real distSqrs[2], real3 directors[2], real invDistProd, real3 forces[3]) {
        real cot = c / s;
        //real dTheta = theta - angleType.theta0;
        //real dCosTheta = cosf(dTheta);



        real a = -angleType.k;

        real a11 = a*c/distSqrs[0];
        real a12 = -a*invDistProd;
        real a22 = a*c/distSqrs[1];

        real b11 = -a*c*cot/distSqrs[0];
        real b12 = a*cot*invDistProd;
        real b22 = -a*c*cot/distSqrs[1];
#ifdef DASH_DOUBLE
        real c0 = cos(angleType.theta0);
        real s0 = cos(angleType.theta0);
#else 
        real c0 = cosf(angleType.theta0);
        real s0 = cosf(angleType.theta0);
#endif
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.theta0);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        forces[0] = (directors[0] * a11 + directors[1] * a12) * c0 + (directors[0] * b11 + directors[1] * b12) * s0;
        forces[1] = (directors[0] * a11 + directors[1] * a12) * -c0 + (directors[0] * b11 + directors[1] * b12) * -s0 
            +
            (directors[1] * a22 + directors[0] * a12) * -c0 + (directors[1] * b22 + directors[0] * b12) * -s0
            ;

        forces[2] = (directors[1] * a22 + directors[0] * a12) * c0 + (directors[1] * b22 + directors[0] * b12) * s0;

        

    }


    inline __device__ real energy(AngleCosineDeltaType angleType, real theta, real3 directors[2]) {
        real dTheta = theta - angleType.theta0;
#ifdef DASH_DOUBLE
        return (1.0 / 3.0) * (1.0 - cos(dTheta)); //energy split between three atoms
#else 
        return (1.0f / 3.0f) * (1.0f - cosf(dTheta)); //energy split between three atoms
#endif
    }
};

#endif

