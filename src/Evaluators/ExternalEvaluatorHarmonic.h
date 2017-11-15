#pragma once

#include "cutils_math.h"

class EvaluatorExternalHarmonic {
	public:
	    real3 k;
        real3 r0;

        // default constructor
        EvaluatorExternalHarmonic () {};
        EvaluatorExternalHarmonic (real3 k_, real3 r0_ ) {
            k = k_;
            r0= r0_;
        };
       
        // force function called by compute_force_external(...) in ExternalEvaluate.h
	inline __device__ real3 force(real3 pos) {
        real3 dr = pos - r0;
        return -k * dr;
        };

        // force function called by compute_energy_external(...) in ExternalEvaluate.h
	inline __device__ real energy(real3 pos) {
        real3 dr  = pos - r0;
		real3 dr2 = dr * dr; 
        return 0.5*(k.x * dr2.x + k.y*dr2.y + k.z*dr2.z) ;
        };

};

