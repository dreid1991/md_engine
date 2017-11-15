#pragma once
#ifndef EVALUATOR_WALL_HARMONIC
#define EVALUATOR_WALL_HARMONIC

#include "cutils_math.h"

class EvaluatorWallHarmonic {
	public:
		real k;
        real r0;

        // default constructor
        EvaluatorWallHarmonic () {};
        EvaluatorWallHarmonic (real k_, real r0_) {
            k = k_;
            r0= r0_;
        };
       
        // setParameters method, called in FixWallHarmonic_temp::prepareForRun()
        //void setParameters(real k_, real r0_) {
         //   k = k_;
         //   r0= r0_;
       // };

        // force function called by compute_wall_iso(...) in WallEvaluate.h
		inline __device__ real3 force(real magProj, real3 forceDir) {
           if (magProj < r0) { 
                real forceScalar = k * (r0 - magProj); 
                return forceDir * forceScalar;
           } else {
                return forceDir * 0.0;
           };

        };


};
#endif

