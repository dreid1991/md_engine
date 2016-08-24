#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"

template <class EVALUATOR, bool COMPUTE_VIRIALS>

__global__ void compute_DPD_iso (*args) {

    
	int idx = GETIDX();
	if (idx < nAtoms) {
		float4 forceWhole = fs[idx];
		uint groupTagAtom = * (uint *) &forceWhole.w;
		// if this atom is assigned to the group affected by this wall fix, then..
		if (groupTagAtom & groupTag) {
			float4 posWhole = xs[idx];
			float3 pos = make_float3(posWhole);
            
            float4 velWhole = vs[idx];
            float3 vel = make_float3(velWhole);
            
            // recall that all of these interactions are pairwise symmetric,
            // or at least functions of the relative velocities
            // consider a way to optimize these?
            //
            
            // we need a loop within each thread covering all other atoms in the neighbor list

            //float3 randForce = eval.randForce(vel1, vel2);
            // we now have our positions and velocities, 
            // let's do something with them
            //


}







