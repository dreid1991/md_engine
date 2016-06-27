#pragma once
#ifndef EVALUATOR_WALL_LJ126
#define EVALUATOR_WALL_LJ126

#include "cutils_math.h"

class EvaluatorWallLJ126 {
	public:
		float sigma;
        float epsilon;
        float r0;

        // default constructor
        EvaluatorWallLJ126 () {};
       
        // setParameters method, called in FixWallLJ126::prepareForRun()
        void setParameters(float sigma_, float epsilon_, float r0_) {
            sigma = sigma_;
            epsilon = epsilon_;
            r0= r0_;
        };
        
        // TODO make this a LJ12-6 potential
        // force function called by compute_wall_iso(...) in WallEvaluate.h
		inline __device__ float3 force(float dist, float magProj, float3 forceDir) {
           // float forceScalar = k * (dist - magProj); 
            return forceDir ;
        };
};
#endif
