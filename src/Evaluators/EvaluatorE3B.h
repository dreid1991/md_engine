#pragma once
#ifndef EVALUATOR_E3B
#define EVALUATOR_E3B

#include "cutils_math.h"
#include "Virial.h" // because we do our computeVirial calls here for threebody forces

void export_EvaluatorE3B();
class EvaluatorE3B {
    public:

        /* Evaluator for E3B3 potential for water; see
         * Craig J. Tainter, Liang Shi, & James L. Skinner, 
         * J. Chem. Theory Comput. 2015, 11, 2268-2277
         * for further details
         */

        // short cutoff for switching function (defaults to 5.0 Angstroms)
        real rs;
        real rstimes3;
        // long cutoff for switching function (defaults to 5.2 Angstroms)
        real rf;
        
        // denominator of the switching function (constant once defined)
        real rfminusrs_cubed_inv;
        // prefactors E2 - pair correction term for two-body TIP4P
        real E2;
        // prefactors A,B,C (units: kJ/mol, in reference paper)
        real Ea;
        real Eb;
        real Ec;

        // the k2 and k3 parameters as presented in E3B3 paper (referenced above)
        // k2 = 4.872 A^-1, k3 = 1.907 A^-1
        real k2;
        real k3;
        real minus_k3; // -1.0 * k3
        
        // default constructor, just to make FixE3B3 happy
        EvaluatorE3B() {};

        // handling of units is addressed in the instantiation of the evaluator in FixE3B3.cu
        // --> so, by now, it is safe to assume that our prefactors & cutoffs are consistent
        //     w.r.t units as the rest of the system
        EvaluatorE3B(real rs_, real rf_, real E2_, 
                      real Ea_, real Eb_, real Ec_,
                      real k2_, real k3_) {
            rs = rs_;
            rf = rf_;
            E2 = E2_;
            Ea = Ea_;
            Eb = Eb_;
            Ec = Ec_;
            k2 = k2_;
            k3 = k3_;
            minus_k3 = -1.0 * k3;

            real rfminusrs = rf_ - rs_;
            rfminusrs_cubed_inv = 1.0 / (rfminusrs * rfminusrs * rfminusrs);
            rstimes3 = 3.0 * rs_;
        };

        // implements the O-O two-body correction to TIP4P/2005
        template <bool COMP_VIRIALS>
        inline __host__ __device__ bool twoBodyForce(const real3 &dr, real3 &fs, Virial &virial) {
#ifdef DASH_DOUBLE
            if (COMP_VIRIALS) {
                real r = length(dr);
                real forceScalar = k2 * E2 * exp(-1.0 * k2 * r) / r;
                real3 force = dr * forceScalar;
                fs += force;
                computeVirial(virial, force, dr);
            } else {
                real r = length(dr);
                real forceScalar = k2 * E2 * exp(-1.0 * k2 * r) / r ;
                fs += (dr * forceScalar);
            }
            return true;
#else
            if (COMP_VIRIALS) {
                real r = length(dr);
                real forceScalar = k2 * E2 * expf(-1.0f * k2 * r) / r;
                real3 force = dr * forceScalar;
                fs += force;
                computeVirial(virial, force, dr);
            } else {
                real r = length(dr);
                real forceScalar = k2 * E2 * expf(-1.0f * k2 * r) / r;
                fs += (dr * forceScalar);
            }
            return true;
#endif
        }

        inline __host__ __device__ real twoBodyEnergy(real r) { return 0.0;}; /* TODO */


        // implements one evaluation of the switching function for smooth cutoff of the potential
        inline __host__ __device__ real switching(real dist) {
            if (dist < rs) {
                return 1.0;
            } 
            if (dist > rf) {
                return 0.0;
            }
        
            real rfMinusDist = rf - dist;
            real rfDistSqr = rfMinusDist * rfMinusDist;
            real sr = (rfDistSqr * (rf - (2.0 * dist) + rstimes3) ) * rfminusrs_cubed_inv;
            return sr;
        }


        // this is f(r) and so must be accounted for in taking the derivative of the potential
        inline __host__ __device__ real dswitchdr(real dist) {
            // should be simple since its a function of the polynomial..
            // -- need to check if this should return 0.0 if outside the range rs < r < rf. be careful here
            if (dist < rs) {
                return 0.0;
            } else if (dist > rf) {
                return 0.0;
            } else {
                // derivative of the polynomial expression w.r.t scalar r_ij scalar.  Returns some real value
                real value = (2.0 * (rf - dist) * (rf - dist)); 
                value -= (2.0 * (rf - dist) * (2.0 * dist + rf - 3.0 * rs) );
                value *= rfminusrs_cubed_inv;
                return value;
            }

        }
        

        // returns [s(r2) * A1 * exp(-A2 (r1 + r2) ) / r1] * (- ds(r1) / dr1 + A2 * s(r1))
        __host__ __device__ real threeBodyForceScalar(real r1, real r2, real preFactor) {
#ifdef DASH_DOUBLE
            return ( (switching(r2) * preFactor * exp(minus_k3 * (r1 + r2)) / r1) * 
                     (k3 * switching(r1) - dswitchdr(r1)));
#else
            return ( (switching(r2) * preFactor * expf(minus_k3 * (r1 + r2)) / r1) *
                     (k3 * switching(r1) - dswitchdr(r1)));
#endif
        }

        template <bool COMP_VIRIALS>
        __host__ __device__ bool threeBodyForce(real3 &fs_a1_sum, real3 &fs_b1_sum, real3 &fs_c1_sum,
                                                Virial &virialsSum_a, Virial &virialsSum_b, Virial &virialsSum_c,

                                                const real3 &r_a1b2, const real3 &r_a1c2, 
                                                const real3 &r_b1a2, const real3 &r_c1a2,

                                                const real3 &r_a1b3, const real3 &r_a1c3,
                                                const real3 &r_b1a3, const real3 &r_c1a3,

                                                const real3 &r_a2b3, const real3 &r_a2c3,
                                                const real3 &r_b2a3, const real3 &r_c2a3) {
                                 

            // get the distance scalars associated with the rij vectors
            real r_a1b2_scalar = length(r_a1b2);  // COMPLETE
            real r_a1c2_scalar = length(r_a1c2);  // COMPLETE

            real r_b1a2_scalar = length(r_b1a2);  // COMPLETE
            real r_c1a2_scalar = length(r_c1a2);  // COMPLETE

            real r_a1b3_scalar = length(r_a1b3);  // COMPLETE
            real r_a1c3_scalar = length(r_a1c3);

            real r_b1a3_scalar = length(r_b1a3);
            real r_c1a3_scalar = length(r_c1a3);

            real r_a2b3_scalar = length(r_a2b3);  // not a force directing vector
            real r_a2c3_scalar = length(r_a2c3);  // not a force directing vector

            real r_b2a3_scalar = length(r_b2a3);  // not a force directing vector
            real r_c2a3_scalar = length(r_c2a3);  // not a force directing vector

            real fs_scalar = 0.0;
            real3 fs_tmp = make_real3(0.0, 0.0, 0.0);

            // local sum; add aggregation to fs sums passed in by reference
            real3 local_a1_sum = make_real3(0.0,0.0,0.0);
            real3 local_b1_sum = make_real3(0.0,0.0,0.0);
            real3 local_c1_sum = make_real3(0.0,0.0,0.0);

            // we group terms according to vector direction, so as to minimize calls to 
            // computeVirials

            /* b1a2 direction terms:
             * f(r_b1a2,r_c1a3)
             * g(b1a2,b3a1)
             * g(b1a2,c3a1)
             * g(c2a3,b1a2)
             * g(b2a3,b1a2)
             * h(b1a2,b3a2)
             * h(b1a2,c3a2)
             */
            
            // f(r_b1a2,r_c1a3); FIRST CALL, set as assignment
            fs_scalar = threeBodyForceScalar(r_b1a2_scalar,r_c1a3_scalar,Ea);
            // g(b1a2,b3a1); AGGREGATE
            fs_scalar += threeBodyForceScalar(r_b1a2_scalar,r_a1b3_scalar,Eb);
            // g(b1a2,c3a1); AGGREGATE
            fs_scalar += threeBodyForceScalar(r_b1a2_scalar,r_a1c3_scalar,Eb);
            // g(c2a3,b1a2); AGGREGATE
            fs_scalar += threeBodyForceScalar(r_b1a2_scalar,r_c2a3_scalar,Eb);
            // g(b2a3,b1a2); AGGREGATE 
            fs_scalar += threeBodyForceScalar(r_b1a2_scalar,r_b2a3_scalar,Eb);
            // h(b1a2,b3a2); AGGREGATE
            fs_scalar += threeBodyForceScalar(r_b1a2_scalar,r_a2b3_scalar,Ec);
            // h(b1a2,c3a2); AGGREGATE
            fs_scalar += threeBodyForceScalar(r_b1a2_scalar,r_a2c3_scalar,Ec);
            
            fs_tmp    = fs_scalar * r_b1a2;
            
            local_b1_sum += fs_tmp;
            
            if (COMP_VIRIALS) {
                computeVirial(virialsSum_b, fs_tmp, r_b1a2);
            }
            
            /* a1b2 direction terms: (b2a1 in E3B1 paper)
             * f(b2a1,c2a3)
             * g(b1a3,b2a1)
             * g(c1a3,b2a1)
             * g(b2a1,b3a2)
             * g(b2a1,c3a2)
             * h(b2a1,b3a1)
             * h(b2a1,c3a1)
             */
            
            // f(r_b2a1,r_c2a3); FIRST CALL, set as assignment
            fs_scalar = threeBodyForceScalar(r_a1b2_scalar,r_c2a3_scalar,Ea);
            // g(b1a3,b2a1)
            fs_scalar += threeBodyForceScalar(r_a1b2_scalar,r_b1a3_scalar,Eb);
            // g(c1a3,b2a1)
            fs_scalar += threeBodyForceScalar(r_a1b2_scalar,r_c1a3_scalar,Eb);
            // g(b2a1,b3a2)
            fs_scalar += threeBodyForceScalar(r_a1b2_scalar,r_a2b3_scalar,Eb);
            // g(b2a1,c3a2)
            fs_scalar += threeBodyForceScalar(r_a1b2_scalar,r_a2c3_scalar,Eb);
            // h(b2a1,b3a1)
            fs_scalar += threeBodyForceScalar(r_a1b2_scalar,r_a1b3_scalar,Ec);
            // h(b2a1,c3a1)
            fs_scalar += threeBodyForceScalar(r_a1b2_scalar,r_a1c3_scalar,Ec);

            fs_tmp    = fs_scalar * r_a1b2;
            local_a1_sum += fs_tmp;
            if (COMP_VIRIALS) {
                computeVirial(virialsSum_a,fs_tmp,r_a1b2);
            }

            /* a1c2 direction: (c2a1 in E3B1 paper) 
             * f(b2a3,c2a1)
             * g(b1a3,c2a1)
             * g(c1a3,c2a1)
             * g(c2a1,b3a2)
             * g(c2a1,c3a2)
             * h(c2a1,b3a1)
             * h(c2a1,c3a1)
             */
           
            // f(b2a3,c2a1)
            fs_scalar =  threeBodyForceScalar(r_a1c2_scalar,r_b2a3_scalar,Ea);
            // g(b1a3,c2a1)
            fs_scalar += threeBodyForceScalar(r_a1c2_scalar,r_b1a3_scalar,Eb);
            // g(c1a3,c2a1)
            fs_scalar += threeBodyForceScalar(r_a1c2_scalar,r_c1a3_scalar,Eb);
            // g(c2a1,b3a2)
            fs_scalar += threeBodyForceScalar(r_a1c2_scalar,r_a2b3_scalar,Eb);
            // g(c2a1,c3a2)
            fs_scalar += threeBodyForceScalar(r_a1c2_scalar,r_a2c3_scalar,Eb);
            // h(c2a1,b3a1)
            fs_scalar += threeBodyForceScalar(r_a1c2_scalar,r_a1b3_scalar,Ec);
            // h(c2a1,c3a1)
            fs_scalar += threeBodyForceScalar(r_a1c2_scalar,r_a1c3_scalar,Ec);

            fs_tmp = fs_scalar * r_a1c2;
            local_a1_sum += fs_tmp;
            if (COMP_VIRIALS) {
                computeVirial(virialsSum_a,fs_tmp,r_a1c2);
            }

            /* c1a2 direction
             * f(b1a3, c1a2)
             * g(c1a2, b3a1)
             * g(c1a2, c3a1)
             * g(b2a3, c1a2)
             * g(c2a3, c1a2)
             * h(c1a2, b3a2)
             * h(c1a2, c3a2)
             */
            // f(b1a3,c1a2)
            fs_scalar =  threeBodyForceScalar(r_c1a2_scalar,r_b1a3_scalar,Ea);
            // g(c1a2,b3a1)
            fs_scalar += threeBodyForceScalar(r_c1a2_scalar,r_a1b3_scalar,Eb);
            // g(c1a2, c3a1)
            fs_scalar += threeBodyForceScalar(r_c1a2_scalar,r_a1c3_scalar,Eb);
            // g(b2a3, c1a2)
            fs_scalar += threeBodyForceScalar(r_c1a2_scalar,r_b2a3_scalar,Eb);
            // g(c2a3, c1a2)
            fs_scalar += threeBodyForceScalar(r_c1a2_scalar,r_c2a3_scalar,Eb);
            // h(c1a2, b3a2)
            fs_scalar += threeBodyForceScalar(r_c1a2_scalar,r_a2b3_scalar,Ec);
            // h(c1a2, c3a2)
            fs_scalar += threeBodyForceScalar(r_c1a2_scalar,r_a2c3_scalar,Ec);

            fs_tmp = fs_scalar * r_c1a2;
            local_c1_sum += fs_tmp;
            if (COMP_VIRIALS) {
                computeVirial(virialsSum_c,fs_tmp,r_c1a2);
            }
            /* a1b3 direction
             * f(b3a1,c3a2)
             * g(b1a2,b3a1)
             * g(c1a2,b3a1)
             * g(b3a1,b2a3)
             * g(b3a1,c2a3)
             * h(b2a1,b3a1)
             * h(c2a1,b3a1)
             */
            // f(b3a1,c3a2)
            fs_scalar =  threeBodyForceScalar(r_a1b3_scalar,r_a2c3_scalar,Ea);
            // g(b1a2,b3a1)
            fs_scalar += threeBodyForceScalar(r_a1b3_scalar,r_b1a2_scalar,Eb);
            // g(c1a2,b3a1)
            fs_scalar += threeBodyForceScalar(r_a1b3_scalar,r_c1a2_scalar,Eb);
            // g(b3a1,b2a3)
            fs_scalar += threeBodyForceScalar(r_a1b3_scalar,r_b2a3_scalar,Eb);
            // g(b3a1,c2a3)
            fs_scalar += threeBodyForceScalar(r_a1b3_scalar,r_c2a3_scalar,Eb);
            // h(b2a1,b3a1)
            fs_scalar += threeBodyForceScalar(r_a1b3_scalar,r_a1b2_scalar,Ec);
            // h(c2a1,b3a1)
            fs_scalar += threeBodyForceScalar(r_a1b3_scalar,r_a1c2_scalar,Ec);
    
            fs_tmp = fs_scalar * r_a1b3;
            local_a1_sum += fs_tmp;
            if (COMP_VIRIALS) {
                computeVirial(virialsSum_a,fs_tmp,r_a1b3);
            }

            /* a1c3 direction
             * f(b3a2,c3a1)
             * g(b1a2,c3a1)
             * g(c1a2,c3a1)
             * g(c3a1,b2a3)
             * g(c3a1,c2a3)
             * h(b2a1,c3a1)
             * h(c2a1,c3a1)
             */
            // f(b3a2,c3a1)
            fs_scalar =  threeBodyForceScalar(r_a1c3_scalar,r_a2b3_scalar,Ea);
            // g(b1a2,c3a1)
            fs_scalar += threeBodyForceScalar(r_a1c3_scalar,r_b1a2_scalar,Eb);
            // g(c1a2,c3a1)
            fs_scalar += threeBodyForceScalar(r_a1c3_scalar,r_c1a2_scalar,Eb);
            // g(c3a1,b2a3)
            fs_scalar += threeBodyForceScalar(r_a1c3_scalar,r_b2a3_scalar,Eb);
            // g(c3a1,c2a3)
            fs_scalar += threeBodyForceScalar(r_a1c3_scalar,r_c2a3_scalar,Eb);
            // h(b2a1,c3a1)
            fs_scalar += threeBodyForceScalar(r_a1c3_scalar,r_a1b2_scalar,Ec);
            // h(c2a1,c3a1)
            fs_scalar += threeBodyForceScalar(r_a1c3_scalar,r_a1c2_scalar,Ec);

            fs_tmp = fs_scalar * r_a1c3;
            local_a1_sum += fs_tmp;
            if (COMP_VIRIALS) {
                computeVirial(virialsSum_a,fs_tmp,r_a1c3);
            }

            /* b1a3 direction
             * f(b1a3,c1a2)
             * g(b1a3,b2a1)
             * g(b1a3,c2a1)
             * g(b3a2,b1a3)
             * g(c3a2,b1a3)
             * h(b1a3,b2a3)
             * h(b1a3,c2a3)
             */
            // f(b1a3,c1a2)
            fs_scalar =  threeBodyForceScalar(r_b1a3_scalar,r_c1a2_scalar,Ea);
            // g(b1a3,b2a1)
            fs_scalar += threeBodyForceScalar(r_b1a3_scalar,r_a1b2_scalar,Eb);
            // g(b1a3,c2a1)
            fs_scalar += threeBodyForceScalar(r_b1a3_scalar,r_a1c2_scalar,Eb);
            // g(b3a2,b1a3)
            fs_scalar += threeBodyForceScalar(r_b1a3_scalar,r_a2b3_scalar,Eb);
            // g(c3a2,b1a3)
            fs_scalar += threeBodyForceScalar(r_b1a3_scalar,r_a2c3_scalar,Eb);
            // h(b1a3,b2a3)
            fs_scalar += threeBodyForceScalar(r_b1a3_scalar,r_b2a3_scalar,Ec);
            // h(b1a3,c2a3)
            fs_scalar += threeBodyForceScalar(r_b1a3_scalar,r_c2a3_scalar,Ec);

            fs_tmp = fs_scalar * r_b1a3;
            local_b1_sum += fs_tmp;
            if (COMP_VIRIALS) {
                computeVirial(virialsSum_b,fs_tmp,r_b1a3);
            }
        
            /* c1a3 direction 
             * f(b1a2,c1a3)
             * g(c1a3,b2a1)
             * g(c1a3,c2a1)
             * g(b3a2,c1a3)
             * g(c3a2,c1a3)
             * h(c1a3,b2a3)
             * h(c1a3,c2a3)
             */
            // f(b1a2,c1a3)
            fs_scalar =  threeBodyForceScalar(r_c1a3_scalar,r_b1a2_scalar,Ea);
            // g(c1a3,b2a1)
            fs_scalar += threeBodyForceScalar(r_c1a3_scalar,r_a1b2_scalar,Eb);
            // g(c1a3,c2a1)
            fs_scalar += threeBodyForceScalar(r_c1a3_scalar,r_a1c2_scalar,Eb);
            // g(b3a2,c1a3)
            fs_scalar += threeBodyForceScalar(r_c1a3_scalar,r_a2b3_scalar,Eb);
            // g(c3a2,c1a3)
            fs_scalar += threeBodyForceScalar(r_c1a3_scalar,r_a2c3_scalar,Eb);
            // h(c1a3,b2a3)
            fs_scalar += threeBodyForceScalar(r_c1a3_scalar,r_b2a3_scalar,Ec);
            // h(c1a3,c2a3)
            fs_scalar += threeBodyForceScalar(r_c1a3_scalar,r_c2a3_scalar,Ec);

            fs_tmp = fs_scalar * r_c1a3;
            local_c1_sum += fs_tmp;
            if (COMP_VIRIALS) {
                computeVirial(virialsSum_c,fs_tmp,r_c1a3);
            }

            // add local sums to fs_sums passed in by reference
            //real3 &fs_a1_sum, real3 &fs_b1_sum, real3 &fs_c1_sum,
            fs_a1_sum += local_a1_sum;
            fs_b1_sum += local_b1_sum;
            fs_c1_sum += local_c1_sum;
            // END OF THREEBODY FORCE COMPUTATION
            return true;
        } // closes threeBodyForce(...) function

};

#endif /* EVALUATOR_E3B */
