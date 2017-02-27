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
        ~ThreeBodyEvaulatorE3B3();

        // handling of units is addressed in the instantiation of the evaluator in FixE3B3.cu
        // --> so, by now, it is safe to assume that our prefactors & cutoffs are consistent
        //     w.r.t units as the rest of the system
        ThreeBodyEvaulatorE3B3(float rs_, float rf_, float E2_, float Ea_, float Eb_, float Ec_) {
            rs = rs_;
            rf = rf_;
            E2 = E2_;
            Ea = Ea_;
            Eb = Eb_;
            Ec = Ec_;
            float rfminusrs = rf_ - rs;
            rfminusrs_cubed_inv = 1.0 / (rfminusrs * rfminusrs * rfminusrs);
        };

        // this implements one force calculation for a given trimer
        //__device__ float3 force(    ) {




        //};


        
        // implements one evaluation of the switching function for smooth cutoff of the potential
        inline __device__ float switching(float dist) {
            if (dist < rs) {
                return 1.0;
            } else if (dist > rf) {
                return 0.0;
            } else {

                float rfminusr1 = rf - dist;
                float rfr1sqr = rfminusr1 * rfminusr1;
                float sr = (rfr1sqr * (rf - (2.0 * dist) + (3.0 * rs) ) ) * rfminusrs_cubed_inv;
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
                // derivative of our polynomial expression
            }
        }

}

#endif /* THREE_BODY_EVALUATOR_E3B3 */
