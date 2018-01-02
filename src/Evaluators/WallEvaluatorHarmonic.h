#pragma once
#ifndef EVALUATOR_WALL_HARMONIC
#define EVALUATOR_WALL_HARMONIC

#include "globalDefs.h"
#include "cutils_math.h"
#include "Vector.h" // this already includes globalDefs

// we export the Evaluators so that they can be explicitly tested via pytest
void export_EvaluatorWallHarmonic();

class EvaluatorWallHarmonic {
	public:
		real k; // spring constant
        real r0;// cutoff
        //real3 storage; // just testing something out
        // default constructor - denote as 'not prepared';
        // -- note that the default ctor is typically never used
        EvaluatorWallHarmonic () {};
        EvaluatorWallHarmonic (real k_, real r0_);

        // force function called by compute_wall_iso(...) in WallEvaluate.h
		inline __host__ __device__ real3 force(real magProj, real3 forceDir) {
           if (magProj < r0) { 
                real forceScalar = k * (r0 - magProj); 
                return forceDir * forceScalar;
           } else {
                return forceDir * 0.0;
           };

        };

        // inline function must remain in header file
        inline __host__ __device__ real energy(real distance) {
            // ok, we have a distance from the wall origin for the dimensions a force is exerted..
            if (distance < r0 ) {
                real distFromCutoff = r0 - distance;
                return 0.5 * k * distFromCutoff * distFromCutoff;

            } else {
                return 0.0;
            }
        };
                                    
        // call 'force' via python interface - for use with test suite
        __host__ Vector forcePy (double distanceFromWall, Vector forceDirection);
        // call 'energy' via python interface - for use with test suite
        __host__ double energyPy (double distance_); 

        // call 'force' via python interface, using device code
        __host__ Vector forcePy_device (double distanceFromWall, Vector forceDirection);

        // call 'energy' via python interface, using device code
        __host__ double energyPy_device (double distance_);


};


#endif

