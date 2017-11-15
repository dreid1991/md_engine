#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"


template <class EVALUATOR, bool COMPUTE_VIRIALS>
__global__ void compute_force_external(int nAtoms,real4 *xs, real4 *fs, uint groupTag,Virial *__restrict__ virials, EVALUATOR eval) 
        {
	int idx = GETIDX();
	if (idx < nAtoms) {
	    real4 forceWhole = fs[idx];
	    uint groupTagAtom = * (uint *) &forceWhole.w;
	    // Check if atom is part of group affected by external potential
	    if (groupTagAtom & groupTag) {
            //Virial virialSum(0, 0, 0, 0, 0, 0);
	        real4 posWhole = xs[idx];
	        real3 pos      = make_real3(posWhole);
            real3 force    = eval.force( pos );      // compute the force due to ext. potential!
            real4 f        = fs[idx];
            f               = f + force;
            fs[idx]         = f;
            //if (COMPUTE_VIRIALS) {
            //    computeVirial(virialSum,force,pos);
            //    virials[idx] += virialSum;
            //}
            }
	    }
	}


template <class EVALUATOR>
__global__ void compute_energy_external(int nAtoms,real4 *xs, real4 *fs, real *perParticleEng, uint groupTag, EVALUATOR eval) 
        {
	int idx = GETIDX();
	if (idx < nAtoms) {
	  real4 forceWhole = fs[idx];
	  uint groupTagAtom = * (uint *) &forceWhole.w;
	  // Check if atom is part of group affected by external potential
	  if (groupTagAtom & groupTag) {
	    real4 posWhole = xs[idx];
	    real3 pos      = make_real3(posWhole);
            real  uext     = eval.energy( pos );      // compute the energy due to ext. potential!
            perParticleEng[idx] += uext;
            }
	  }
	}

