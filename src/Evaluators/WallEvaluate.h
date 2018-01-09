#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"

template <class EVALUATOR, bool COMPUTE_VIRIALS>
__global__ void compute_wall_iso(int nAtoms,real4 * __restrict__ xs, 
                                 real4 * __restrict__ fs,
         Virial *__restrict__ virials, 
         real3 wall,
		real3 forceDir,  uint groupTag, EVALUATOR eval) {


	int idx = GETIDX();
	if (idx < nAtoms) {
		real4 forceWhole = fs[idx];
		uint groupTagAtom = * (uint *) &forceWhole.w;
		// if this atom is assigned to the group affected by this wall fix, then..
		if (groupTagAtom & groupTag) {
			real4 posWhole = xs[idx];
			real3 pos = make_real3(posWhole);
			real3 particleDisplacement = pos - wall;
			real projection = dot(particleDisplacement, forceDir);
			real magProj = cu_abs(projection);
            real3 force = eval.force(magProj, forceDir);

            real4 f = fs[idx];
            if (projection >= 0) {
                f = f + force;
            } else {
                printf("Atom pos %f %f %f wall %f %f %f\n", pos.x, pos.y, pos.z, wall.x, wall.y, wall.z);
                assert(projection>0); // projection should be greater than 0, otherwise
                // the wall is ill-defined (forceDir pointing out of box)
            }
            fs[idx] = f;

            if (COMPUTE_VIRIALS) {

                Virial virialsSum = Virial(0, 0, 0, 0, 0, 0);
                computeVirial(virialsSum, force, particleDisplacement);
                virials[idx] += virialsSum;

            }

		}
	}
}

template<class EVALUATOR>
__global__ void compute_wall_energy(int nAtoms, 
                                    real4 *xs, 
                                    real *perParticleEng, 
                                    real4 *fs, // for groupTag
                                    real3 wall,
                                    real3 forceDir,
                                    uint groupTag,
                                    EVALUATOR eval) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        real4 forceWhole = fs[idx];
		uint groupTagAtom = * (uint *) &forceWhole.w;
		// if this atom is assigned to the group affected by this wall fix, then..
		if (groupTagAtom & groupTag) {

            real4 posWhole = xs[idx];
            real3 pos = make_real3(posWhole);

            // get the distance from the wall
			real3 particleDisplacement = pos - wall;
            // gets the distance in the dimensions that matter
			real projection = dot(particleDisplacement, forceDir);

            real magnitudeOfProjection = cu_abs(projection);
            
            real energy = eval.energy(magnitudeOfProjection);
            
            perParticleEng[idx] += energy;
        }
    }
}
    


