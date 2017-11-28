#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"


template <class EVALUATOR, bool COMPUTE_VIRIALS>
__global__ void compute_wall_iso(int nAtoms,real4 *xs, real4 *fs,real3 origin,
		real3 forceDir,  uint groupTag, EVALUATOR eval) {


	int idx = GETIDX();
	if (idx < nAtoms) {
		real4 forceWhole = fs[idx];
		uint groupTagAtom = * (uint *) &forceWhole.w;
		// if this atom is assigned to the group affected by this wall fix, then..
		if (groupTagAtom & groupTag) {
			real4 posWhole = xs[idx];
			real3 pos = make_real3(posWhole);
			real3 particleDist = pos - origin;
			real projection = dot(particleDist, forceDir);
			real magProj = cu_abs(projection);
            real3 force = eval.force(magProj, forceDir);

            real4 f = fs[idx];
            if (projection >= 0) {
                f = f + force;
            } else {
                printf("Atom pos %f %f %f wall origin %f %f %f\n", pos.x, pos.y, pos.z, origin.x, origin.y, origin.z);
                assert(projection>0); // projection should be greater than 0, otherwise
                // the wall is ill-defined (forceDir pointing out of box)
            }
            fs[idx] = f;


			 
		}
	}
}





