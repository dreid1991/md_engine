#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"
#include "Vector.h"
#include <cmath>

class EvaluatorDPD_T {
public:

    float sigma;
    float gamma;
    float rcut;
    float  s;

    // inverse square root of the current dt, used when calculating the contribution of the thermal noise
    float invSqrtdt;

    // our constructors
    EvaluatorDPD_T() {};


    // sigma is the thermal noise amplitude; gamma is the friction factor
    // rcut is the cutoff of the weight function;
    // s defines the exponent in the weight function (usually 1, sometimes 2, etc... )
    // cast as float to avoid any ambiguity of floating point math operations
    EvaluatorDPD_T (float sigma_, float gamma_, float rcut_, float s_) {
        sigma = sigma_;
        gamma = gamma_;
        rcut = rcut_;
        s = s_;
    };

    void updateGamma(float gamma_) {
        gamma = gamma_;
    };

    void updateSigma(float sigma_) {
        sigma = sigma_;
    };
    // desired syntax: wRij = weightRandom(dist) (exponent s should be provided in construction)
    inline __device__  float3 weightRandom(dist) {
        float weightFactor = pow((1.0f - (dist / rcut)) , s);
        return weightFactor;
        // weight dissipative is just wRij ** 2
    };

    // this is a function because only the dissipative forces are computed in the stepFinal compute call
    inline __device__ float3 forceDissipative(float weight, float3 vel, float3 othervel, float3 eij) { 
        // compute the dissipative force given two velocities, wDij, and the direction vector eij
        float3 relativeVelocity = vel - othervel;
        float3 force = gamma * weight * (dotProduct(relativeVelocity,eij)) * eij; // TODO dotProduct syntax
        // and verify that we are doing the matrix multiplication correctly (and not collapsing any dimensions)
    };


    // this is the force function called during force() from the integrator. stepFinal calls only the above function
    inline __device__ float3 force (float3 pos, float3 otherpos, float3 vel, float3 othervel, 
                                    float3 &forces_dissipative, int timestep, int seed1, int seed2) {
        // compute the distance between the atoms and determine if they are inside of the cutoff radius
        // if not, set forces_dissipative to (0.0f, 0.0f, 0.0f) and return force = (0.0f, 0.0f, 0.0f)
        // else, continue with the routine
        // if so, compute the random weight factor; from this, compute dissipative weight factor
        if (dist < rcut) {
            float wRij = weightFactor(dist);
            float wDij = wRij * wRij;

            // compute the normalized direction vector between atoms i and j
            float3 eij = (pos - otherpos) / dist; 

            // get some random numbers from Saru microstream
            //float3 randomVectorij = Saru(seed1, seed2, timestep); // TODO correct this syntax (currently pseudocode)
            // sqrtdt can be instantiated during prepareForRun()
            float3 fRij = wRij * (randomVectorij) * sigma * eij / sqrtdt;
            float3 fDij = forceDissipative(wDij, vel, othervel, eij)  ;
            forces_dissipative = fDij;
            float3 force = fDij + fRij;
            return force;
        } else {
            forces_dissipative = (0.0f, 0.0f, 0.0f)
                return (0.0f, 0.0f, 0.0f);
        };
        // then call Saru and compute FRij
        // then compute the relative velocities (careful othe directionality! computing vij, not vji
        // set forces_dissipative to the computed dissipative forces, 
        // add fRij/sqrtdt and fDij, return in the forces vector
    };


};


// ideally this template will serve as a basis for a general DPD compute function; for now, its only for isothermal DPD
template <class EVALUATOR>
__global__ void computeDPD_Isothermal (int nAtoms, const float4 *__restrict__ xs, float4 *__restrict__ fs, const float4 *__restrict__ vs, 
                                       const uint *__restrict__ ids, float3 *__restrict__ fds, const uint16_t *__restrict__ neighborCounts, 
                                       const uint *__restrict__ neighborList, const uint32_t * __restrict__ cumulSumMaxPerBlock,
                                       int warpSize, int timestep,  BoundsGPU bounds,  const uint __restrict__ groupTag,  
                                       EVALUATOR eval) {

    // if we are only using this for isothermal dpd, do we need to pass in the evaluator via templating? 
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 forceWhole = fs[idx];

        fds[idx] = (0.0, 0.0, 0.0);

        float3 forceSum = make_float3(0, 0, 0);

        uint groupTagAtom = * (uint *) &forceWhole.w;

        int baseSeed1 = ids[idx];

        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);

        if (groupTagAtom & groupTag) {

            float4 posWhole = xs[idx];
            float3 pos = make_float3(posWhole);

            float4 velWhole = vs[idx];
            float3 vel = make_float3(velWhole);

            float3 forces = make_float3(forceWhole);

            float3 forces_dissipative = fds[idx];

            int numNeigh = neighborCounts[idx];
            for (int j=0; j<numNeigh; j++) {

                int nlistIdx = baseIdx + warpSize * j;
                uint otherIdx = (neighborList[nlistIdx]) & EXCL_MASK;
                int baseSeed2 = ids[otherIdx];

                float4 otherPosWhole = xs[otherIdx];
                float3 otherPos = make_float3(otherPosWhole);

                float4 otherVelWhole = vs[otherIdx];
                float3 otherVel = make_float3(otherVelWhole);

                if (baseSeed1 < baseSeed2) {
                    int seed1 = baseSeed1;
                    int seed2 = baseSeed2; 
                } else {
                    int seed1 = baseSeed2;
                    int seed2 = baseSeed1; 
                };

                // note that forces_dissipative is passed by reference to the function - we modify it 
                forces = eval.force(pos, otherpos, vel, othervel, forces_dissipative,timestep,seed1, seed2);
                forceSum += forces;
                // separately track the dissipative forces (should we track the random ones as well,
                // since they have to be modified by 1/(sqrt(dt)) prior to being passed to forceSum?
                // also do we not want to do this fds[idx] += every loop ? unless the compiler
                // constructed a tmp copy, we're doing excessive calls to the outside function
                fds[idx] += forces_dissipative;
            };

            // we have now looped over the neighbor list. what do
        };
    };
};







