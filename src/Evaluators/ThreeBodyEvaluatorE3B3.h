#pragma once
#ifndef THREE_BODY_EVALUATOR_E3B3
#define THREE_BODY_EVALUATOR_E3B3

#include "cutils_math.h"

class ThreeBodyEvaluatorE3B3 {
    public:

        /* Evaluator for E3B3 potential for water; see
         * Craig J. Tainter, Liang Shi, & James L. Skinner, 
         * J. Chem. Theory Comput. 2015, 11, 2268-2277
         * for further details
         */

        // short cutoff for switching function (defaults to 5.0 Angstroms)
        float rs;
        float rstimes3;
        // long cutoff for switching function (defaults to 5.2 Angstroms)
        float rf;
        
        // denominator of the switching function (constant once defined)
        float rfminusrs_cubed_inv;
        // prefactors E2 - pair correction term for two-body TIP4P
        float E2;
        // prefactors A,B,C (units: kJ/mol, in reference paper)
        float Ea;
        float Eb;
        float Ec;

        // delete our default constructor
        ~ThreeBodyEvaluatorE3B3();

        // handling of units is addressed in the instantiation of the evaluator in FixE3B3.cu
        // --> so, by now, it is safe to assume that our prefactors & cutoffs are consistent
        //     w.r.t units as the rest of the system
        ThreeBodyEvaluatorE3B3(float rs_, float rf_, float E2_, float Ea_, float Eb_, float Ec_) {
            rs = rs_;
            rf = rf_;
            E2 = E2_;
            Ea = Ea_;
            Eb = Eb_;
            Ec = Ec_;
            float rfminusrs = rf_ - rs;
            rfminusrs_cubed_inv = 1.0 / (rfminusrs * rfminusrs * rfminusrs);
            rstimes3 = 3.0 * rs_;
        };

        // implements one evaluation of the switching function for smooth cutoff of the potential
        inline __device__ float switching(float dist) {
            if (dist < rs) {
                return 1.0;
            } else if (dist > rf) {
                return 0.0;
            } else {
                // rf
                float rfMinusDist = rf - dist;
                float rfDistSqr = rfMinusDist * rfMinusDist;
                float sr = (rfDistSqr * (rf - (2.0 * dist) + rstimes3) ) * rfminusrs_cubed_inv;
                return sr;
            }
        }


        // this is f(r) and so must be accounted for in taking the derivative of the potential
        inline __device__ float3 dswitchdr(float dist) {
            // should be simple since its a function of the polynomial..
            // -- need to check if this should return 0.0 if outside the range rs < r < rf. be careful here
            if (dist < rs) {
                return 0.0;
            } else if (dist > rf) {
                return 0.0;
            } else {
                // derivative of the polynomial expression w.r.t scalar r_ij magnitude.  Returns some float value
                float value = (2.0 * (rf - dist) * (rf - dist)); 
                value -= (2.0 * (rf - dist) * (2.0 * dist + rf - 3.0 * rs) );
                value *= rfminusrs_cubed_inv;
                return value;
            }

        }
        

        inline __device__ float threeBodyForceScalar(float magnitude) {
            return switching(magnitude) * expf(-k3 * magnitude);
        }

        inline __device__ float3 threeBodyInteraction(float rij_scalar, float3 rij, float scaleR2, float scaleR1) {
            // rij_scalar denotes the magnitude of vector rij
            // rij is the vector from atom i to atom j
            // scaleR2 is the aggregate product of some other distance with its switch factor value
            
            // and later we will multiply this by the scalar multiplier associated with the contribution 
            // -- i.e., type A, type B, or type C
            // TODO
            // float3 forceToReturn = -1.0 * dswitchdr(rij_scalar) * scaleR2 * expf(-k3 * 



        }
        // pass fs_a1_sum by reference b/c we return it to the function modified

        /* old notation (used oxygen as reference atom rather than hydrogens, as we should
        __device__ void threeBodyForce(float3 &fs_a1_sum, float3 &fs_b1_sum, float3 &fs_c1_sum,
                                       float fs_a1b2_scalar, float fs_a1c2_scalar,
                                       float fs_a2b1_scalar, float fs_a2c1_scalar,
                                       float3 r_a1b2, float r_a1b2_magnitude,
                                       float3 r_a1c2, float r_a1c2_magnitude,
                                       float3 r_a1b3, float r_a1b3_magnitude,
                                       float3 r_a1c3, float r_a1c3_magnitude,
                                       float3 r_a2b1, float r_a2b1_magnitude,
                                       float3 r_a2c1, float r_a2c1_magnitude,
                                       float3 r_a2b3, float r_a2b3_magnitude,
                                       float3 r_a2c3, float r_a2c3_magnitude,
                                       float3 r_a3b1, float r_a3b1_magnitude,
                                       float3 r_a3c1, float r_a3c1_magnitude, 
                                       float3 r_a3b2, float r_a3b2_magnitude, 
                                       float3 r_a3c2, float r_a3c2_magnitude) {

        */
        __device__ void threeBodyForce(float3 &fs_a1_sum, float3 &fs_b1_sum, float3 &fs_c1_sum,
                                        float fs_b2a1_scalar, float fs_c2a1_scalar,
                                        float fs_b1a2_scalar, float fs_c1a2_scalar,
                                        float3 r_b2a1, float r_b2a1_magnitude,
                                        float3 r_c2a1, float r_c2a1_magnitude,
                                        float3 r_b3a1, float r_b3a1_magnitude,
                                        float3 r_c3a1, float r_c3a1_magnitude,
                                        float3 r_b1a2, float r_b1a2_magnitude,
                                        float3 r_c1a2, float r_c1a2_magnitude,
                                        float3 r_b3a2, float r_b3a2_magnitude,
                                        float3 r_c3a2, float r_c3a2_magnitude,
                                        float3 r_b1a3, float r_b1a3_magnitude,
                                        float3 r_c1a3, float r_c1a3_magnitude, 
                                        float3 r_b2a3, float r_b2a3_magnitude, 
                                        float3 r_c2a3, float r_c2a3_magnitude);
            // so, first, we compute the force scalars associated with each of the 12 unique intermolecular OH distances
            // -- note: we already passed in a1b2, a1c2, a2b1, and a2c1 evaluations of the exp(-k3 r) value
            // get the a1b3 and a1c3 terms
            float fs_b3a1_scalar = threeBodyForceScalar(r_b3a1_magnitude);
            float fs_c3a1_scalar = threeBodyForceScalar(r_c3a1_magnitude);
        
            // get the a2b3 and a2c3 terms
            float fs_b3a2_scalar = threeBodyForceScalar(r_b3a2_magnitude);
            float fs_c3a2_scalar = threeBodyForceScalar(r_c3a2_magnitude);

            // get the a3b1, a3c1, a3b2, and a3c2 terms
            float fs_b1a3_scalar = threeBodyForceScalar(r_b1a3_magnitude);
            float fs_c1a3_scalar = threeBodyForceScalar(r_c1a3_magnitude);
            float fs_b2a3_scalar = threeBodyForceScalar(r_b2a3_magnitude);
            float fs_c2a3_scalar = threeBodyForceScalar(r_c2a3_magnitude);



            /* old notation
            // get the a1b3 and a1c3 terms
            float fs_a1b3_scalar = threeBodyForceScalar(r_a1b3_magnitude);
            float fs_a1c3_scalar = threeBodyForceScalar(r_a1c3_magnitude);
        
            // get the a2b3 and a2c3 terms
            float fs_a2b3_scalar = threeBodyForceScalar(r_a2b3_magnitude);
            float fs_a2c3_scalar = threeBodyForceScalar(r_a2c3_magnitude);

            // get the a3b1, a3c1, a3b2, and a3c2 terms
            float fs_a3b1_scalar = threeBodyForceScalar(r_a3b1_magnitude);
            float fs_a3c1_scalar = threeBodyForceScalar(r_a3c1_magnitude);
            float fs_a3b2_scalar = threeBodyForceScalar(r_a3b2_magnitude);
            float fs_a3c2_scalar = threeBodyForceScalar(r_a3c2_magnitude);

            */
            // make local force sum vectors for the three atoms
            float3 local_a1_force_sum = make_float3(0.0, 0.0, 0.0);
            float3 local_b1_force_sum = make_float3(0.0, 0.0, 0.0);
            float3 local_c1_force_sum = make_float3(0.0, 0.0, 0.0);
            
            // now, compute the terms unique to a1, the oxygen atom on our reference molecule.
            // ---- first, the A-type anti-cooperative contributions that are unique to a1
            

            // compute the scalar factors for the shared terms (see KS 2008 to see what terms are shared by a1, b1, c1)
            

        } // end threeBodyForce(*bigArgsListHere)
                                        
        // and we need separate functions to calculate the A,B,C force contributions

}

#endif /* THREE_BODY_EVALUATOR_E3B3 */
