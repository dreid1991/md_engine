#pragma once
#ifndef EVALUATOR_WALL_LJ126
#define EVALUATOR_WALL_LJ126

#include "cutils_math.h"

void export_EvaluatorWallLJ126();

class EvaluatorWallLJ126 {
	public:
		real sigma;
        real epsilon;
        real epsilonTimes4;
        real epsilonTimes24;
        real r0;
        
        real sig2;
        real sig6;
        real sig12;
 
        real engShift; // energy at cutoff; we just calculate this on instantiation
        // default constructor
        EvaluatorWallLJ126 () {};
        
        // another constructor here
        EvaluatorWallLJ126(real sigma_, real epsilon_, real r0_); 

        // force function called by compute_wall_iso(...) in WallEvaluate.h
		inline __host__ __device__ real3 force(real magProj, real3 forceDir) {
            if (magProj < r0) {
                real r_inv = 1.0/magProj;
                real r2_inv = r_inv * r_inv;
                real r6_inv = r2_inv * r2_inv * r2_inv;
                real forceScalar = r6_inv * r_inv * ( ( 2.0 * epsilonTimes24 * sig12 * r6_inv - epsilonTimes24 * sig6));
                return forceDir * forceScalar ;
            } else {
                return forceDir * 0.0 ;
            };
        };

        // energy is shifted so that it is 0 at the cutoff
        inline __host__ __device__ real energy(real distance) {
            if (distance <= r0) {
                
                // sig2 / r2
                real mult2  = sig2 / (distance * distance);
                // sig6 / r6
                real mult6  = mult2 * mult2 * mult2; 
                // sig12 / r12
                real mult12 = mult6 * mult6;

                real result = epsilonTimes4 * (mult12 - mult6) + engShift;
                return result;
            } else {
                return 0.0;
            }

        }

        // python interface that calls force function
        __host__ Vector forcePy(double magProj, Vector forceDir);

        // python interface that calls the energy function
        __host__ double energyPy(double distance_);

        // python interface that calls force function
        __host__ Vector forcePy_device(double magProj, Vector forceDir);

        // python interface that calls the energy function
        __host__ double energyPy_device(double distance_);

};
#endif
