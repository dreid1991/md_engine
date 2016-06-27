#pragma once
#ifndef EVALUATOR_WALL_HARMONIC
#define EVALUATOR_WALL_HARMONIC

#include "cutils_math.h"

class EvaluatorWallHarmonic {
	public:
		float k;
        float r0;

        // default constructor
        EvaluatorWallHarmonic () {};
       
        // setParameters method, called in FixWallHarmonic_temp::prepareForRun()
        void setParameters(float k_, float r0_) {
            k = k_;
            r0= r0_;
        };

        // force function called by compute_wall_iso(...) in WallEvaluate.h
		inline __device__ float3 force(float dist, float magProj, float3 forceDir) {
            float forceScalar = k * (dist - magProj); 
            return forceDir * forceScalar;
        };
};
#endif

