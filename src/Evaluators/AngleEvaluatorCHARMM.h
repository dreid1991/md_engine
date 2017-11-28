#pragma once
#ifndef EVALUATOR_ANGLE_CHARMM
#define EVALUATOR_ANGLE_CHARMM

#include "cutils_math.h"
#include "Angle.h"
#define EPSILON 0.00001f
class AngleEvaluatorCHARMM {
public:

    //evaluator.force(theta, angleType, s, distSqrs, directors, invDotProd);
    inline __device__ real3 force(AngleCHARMMType angleType, real theta, real s, real c, real distSqrs[2], real3 directors[2], real invDistProd, int myIdxInAngle) {
        real dTheta = theta - angleType.theta0;
        real forceConst = angleType.k * dTheta;
        real a     = -forceConst * s;
        real a11   = a*c/distSqrs[0];
        real a12   = -a*invDistProd;
        real a22   = a*c/distSqrs[1];

        // Added code for computing Urey-Bradley component (MW)
        real3 dr31;
        dr31        = directors[1] - directors[0]; // Urey-Bradley bond between 1 and 3 atoms
        real rsq31 = dot(dr31,dr31);
#ifdef DASH_DOUBLE
        real r31   = sqrt(rsq31);
#else 
        real r31   = sqrtf(rsq31);
#endif
        real drub  = r31 - angleType.rub;
        real rk    = angleType.kub * drub;
        real fub   = -rk / r31;   // consider safe-checking for r31 > 0.0 ?
        //printf("Eang= %f\n", (1.0f / 2.0f) *( dTheta * dTheta * angleType.k + drub * drub * angleType.kub ));
        // End added code
        
        if (myIdxInAngle==0) {
            return (directors[0] * a11) + (directors[1] * a12) - dr31 * fub ;
        } else if (myIdxInAngle==1) {
            return ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12))*-1.0f ; 
        } else {
            return (directors[1] * a22) + (directors[0] * a12) + dr31 * fub ;
        }


    }



    inline __device__ void forcesAll(AngleCHARMMType angleType, real theta, real s, real c, real distSqrs[2], real3 directors[2], real invDistProd, real3 forces[3]) {
        real dTheta = theta - angleType.theta0;
        //   printf("current %f theta eq %f idx %d, type %d\n", acosf(c), angleType.theta0, myIdxInAngle, type);
        

        real forceConst = angleType.k * dTheta;
        real a = - forceConst * s;
        real a11 = a*c/distSqrs[0];
        real a12 = -a*invDistProd;
        real a22 = a*c/distSqrs[1];
        // Added code for computing Urey-Bradley component (MW)
        real3 dr31;
        dr31        = directors[1] - directors[0]; // Urey-Bradley bond between 1 and 3 atoms
        real rsq31 = dot(dr31,dr31);
#ifdef DASH_DOUBLE
        real r31   = sqrt(rsq31);
#else
        real r31   = sqrtf(rsq31);
#endif
        real drub  = r31 - angleType.rub;
        real rk    = angleType.kub * drub;
        real fub   = -rk / r31;   // consider safe-checking for r31 > 0.0 ?
        // End added code
        forces[0] = (directors[0] * a11) + (directors[1] * a12) - dr31 * fub ;
        forces[1] = ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * -1.0 ; 
        forces[2] = (directors[1] * a22) + (directors[0] * a12) + dr31 * fub ;


    }



    inline __device__ real energy(AngleCHARMMType angleType, real theta, real3 directors[2]) {
        real dTheta = theta - angleType.theta0;
        real3 dr31;
        dr31        = directors[1] - directors[0]; // Urey-Bradley bond between 1 and 3 atoms
        real rsq31 = dot(dr31,dr31);
#ifdef DASH_DOUBLE
        real r31   = sqrt(rsq31);
#else
        real r31   = sqrtf(rsq31);
#endif
        real drub  = r31 - angleType.rub;
//        printf("theta = %f, r31 = %f, theta = %f,k = %f,rub = %f, kub = %f\n",theta,r31,angleType.theta0,angleType.k,angleType.rub,angleType.kub);
//        printf("eang = %f\n", 0.5f *( dTheta * dTheta * angleType.k + drub * drub * angleType.kub ) );
#ifdef DASH_DOUBLE
        return (1.0 / 6.0) *( dTheta * dTheta * angleType.k + drub * drub * angleType.kub ) ; // 1/6 comes from 1/3 (energy split between three atoms) and 1/2 from 1/2 k dtheta^2
#else
        return (1.0f / 6.0f) *( dTheta * dTheta * angleType.k + drub * drub * angleType.kub ) ; // 1/6 comes from 1/3 (energy split between three atoms) and 1/2 from 1/2 k dtheta^2
#endif

    }
};

#endif

