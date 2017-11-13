#pragma once
#ifndef EVALUATOR_WALL_LJ126
#define EVALUATOR_WALL_LJ126

#include "cutils_math.h"

class EvaluatorWallLJ126 {
	public:
		real sigma;
        real epsilonTimes24;
        real r0;
        
        real sig2;
        real sig6;
        real sig12;
        
        // default constructor
        EvaluatorWallLJ126 () {};
        
        // another constructor here
        EvaluatorWallLJ126(real sigma_, real epsilon_, real r0_) {
            sigma = sigma_;
            epsilonTimes24 = 24.0 * epsilon_;
            r0 = r0_;
            sig2 = sigma_ * sigma_;
            sig6 = sig2 * sig2 * sig2;
            sig12 = sig6 * sig6;
        }; 
        // force function called by compute_wall_iso(...) in WallEvaluate.h
		inline __device__ real3 force(real magProj, real3 forceDir) {
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
};
#endif
