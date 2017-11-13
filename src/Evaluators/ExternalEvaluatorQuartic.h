#pragma once

#include "cutils_math.h"

class EvaluatorExternalQuartic {
	public:
	real3 k1;
	real3 k2;
	real3 k3;
	real3 k4;
        real3 r0;

        // default constructor
        EvaluatorExternalQuartic () {};
        EvaluatorExternalQuartic (real3 k1_, real3 k2_, real3 k3_, real3 k4_, real3 r0_ ) {
            k1 = k1_;
            k2 = k2_;
            k3 = k3_;
            k4 = k4_;
            r0 = r0_;
        };
       
        // force function called by compute_force_external(...) in ExternalEvaluate.h
	inline __device__ real3 force(real3 pos) {
                real3 dr  = pos - r0;
		real3 dr2 = dr * dr;
		real3 dr3 = dr2* dr; 
        	return -k1 - 2*k2*dr -3*k3*dr2 - 4*k4*dr3;
        };

        // force function called by compute_energy_external(...) in ExternalEvaluate.h
	inline __device__ real energy(real3 pos) {
                real3 dr  = pos - r0;
		real3 dr2 = dr * dr;
		real3 dr3 = dr2* dr; 
		real3 dr4 = dr2* dr2; 
        	return dot(k1,dr) + dot(k2,dr2) + dot(k3,dr3) + dot(k4,dr4);
        };

};

