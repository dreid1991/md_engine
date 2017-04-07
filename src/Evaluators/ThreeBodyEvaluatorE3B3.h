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

        // this implements one force calculation for a given trimer
        //__device__ float3 force(    ) {


        //};
       
        // bounds.minImage gets the minimum image vector between two atoms, so we'll pass all that info here
        __device__ bool isValidTrimer( *args ){

            return true;

        }
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
                // derivative of our polynomial expression w.r.t. r... is this a vector?
            }
        }

        __device__ void threeBodyForce(float3 &fs_a1_sum, float3 &fs_b1_sum, float3 &fs_c1_sum,
                                       float3 r_a1b2, float r_a1b2_magnitude,
                                       float3 r_a1c2, float r_a1c2_magntiude,
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





        } // end threeBodyForce(*bigArgsListHere)
                                        
        // and we need separate functions to calculate the A,B,C force contributions

}

#endif /* THREE_BODY_EVALUATOR_E3B3 */
