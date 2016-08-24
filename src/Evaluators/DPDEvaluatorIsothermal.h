#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"

template <class EVALUATOR, bool COMPUTE_VIRIALS>

__global__ void computeDPD_Isothermal (int nAtoms, const float4 *__restrict__ xs, float4 *__restrict__ fs, const float4 *__restrict__ vs, 
                                       const uint *__restrict__ ids, float4 *__restrict__ fds, const uint16_t *__restrict__ neighborCounts, 
                                       const uint *__restrict__ neighborList, int warpSize,int timestep,  BoundsGPU bounds, 
                                       Virial *__restrict__ virials, EVALUATOR eval) {

    // if we are only using this for isothermal dpd, do we need to pass in the evaluator via templating? 
    // also, what are we doing with the virials in this fix?
	int idx = GETIDX();
	if (idx < nAtoms) {
    // if we're looking at an atom,
        // we need our fs, vs, xs, fds, neighbor list, ids, and timestep
		float4 forceWhole = fs[idx];
        
        // zero the virials sum
        Virial virialsSum = Virial(0, 0, 0, 0, 0, 0);

        // initialize the force sum of this fix
        float3 forceSum = make_float3(0, 0, 0);

        // verify that this fix applies; 
		uint groupTagAtom = * (uint *) &forceWhole.w;

        // determine the atom index of the atom of interest in this thread
        int baseSeed1 = ids[idx];

        // but we also need to access the neighbor list of said atom
        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);

		// if this atom is assigned to the group affected by this fix, then..
		if (groupTagAtom & groupTag) {

            // extract the positions, velocities, forces, and dissipative forces
			float4 posWhole = xs[idx];
			float3 pos = make_float3(posWhole);
            
            float4 velWhole = vs[idx];
            float3 vel = make_float3(velWhole);
            
            float4 forces = make_float3(forceWhole);
            
            // we track the dissipative forces separately, as they will be updated again in stepFinal();
            float3 forces_dissipative = fds[idx];

            int numNeigh = neighborCounts[idx];
            for (int j=0; j<numNeigh; j++) {

                 // now we need to get the order of seed1, seed2, and send this to the 
                 // dissipative and random evaluators, respectively; then add the computed forces
                 // to the forceSum
                 // first, though, compute the relative distance and director vector (after determining which
                 // atom has a lower atom index!) because these will be used by both F_Dissipative and F_Random
                 int nlistIdx = baseIdx + warpSize * j;
                 //uint otherIdxRaw = neighborList[nlistIdx];
                 // one step, because we don't need the multiplier for this fix (see if this works!)
                 uint otherIdx = (neighborList[nlistIdx]) & EXCL_MASK;
                 // uint otherIdx = otherIdxRaw & EXCL_MASK;
                 int baseSeed2 = ids[otherIdx];
                 
                 // get the positions and velocities of the neighboring particle 
                 float4 otherPosWhole = xs[otherIdx];
                 float3 otherPos = make_float3(otherPosWhole);

                 float4 otherVelWhole = vs[otherIdx];
                 float3 otherVel = make_float3(otherVelWhole);
           
                 // note that we do not care about the forces (dissipative or otherwise) pertaining to 
                 // particle j; these will be calculated by another thread
                
                 float3 




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







