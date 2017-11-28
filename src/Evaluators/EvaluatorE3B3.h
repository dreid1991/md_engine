#pragma once
#ifndef EVALUATOR_E3B3
#define EVALUATOR_E3B3

#include "cutils_math.h"

class EvaluatorE3B3 {
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
        
        // default constructor, just to make FixE3B3 happy
        EvaluatorE3B3() {};

        // handling of units is addressed in the instantiation of the evaluator in FixE3B3.cu
        // --> so, by now, it is safe to assume that our prefactors & cutoffs are consistent
        //     w.r.t units as the rest of the system
        EvaluatorE3B3(real rs_, real rf_, real E2_, 
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

            real rfminusrs = rf_ - rs_;
            rfminusrs_cubed_inv = 1.0 / (rfminusrs * rfminusrs * rfminusrs);
            rstimes3 = 3.0 * rs_;
        };
        inline __device__ int getNumberWithinCutoff(real d1, real d2, real d3, real d4) {
            int count = 0;
            if (d1 < rf) count += 1;
            if (d2 < rf) count += 1;
            if (d3 < rf) count += 1;
            if (d4 < rf) count += 1;
            return count;
        }

        // implements the O-O two-body correction to TIP4P/2005
        inline __device__ real3 twoBodyForce(real3 dr, real r) {
#ifdef DASH_DOUBLE
            real forceScalar = k2 * E2 * exp(-1.0 * k2 * r) / r;
#else
            real forceScalar = k2 * E2 * expf(-1.0f * k2 * r) / r;
#endif
            return dr * forceScalar;
        }

        // implements one evaluation of the switching function for smooth cutoff of the potential
        inline __device__ real switching(real dist) {
            if (dist < rs) {
                return 1.0;
            } else if (dist > rf) {
                return 0.0;
            } else {
                // rf
                real rfMinusDist = rf - dist;
                real rfDistSqr = rfMinusDist * rfMinusDist;
                real sr = (rfDistSqr * (rf - (2.0 * dist) + rstimes3) ) * rfminusrs_cubed_inv;
                return sr;
            }
        }


        // this is f(r) and so must be accounted for in taking the derivative of the potential
        inline __device__ real dswitchdr(real dist) {
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
        

        inline __device__ real threeBodyForceScalar(real magnitude) {
#ifdef DASH_DOUBLE
            return switching(magnitude) * exp(-1.0 * k3 * magnitude);
#else
            return switching(magnitude) * expf(-1.0f * k3 * magnitude);
#endif
        }

        inline __device__ real3 threeBodyInteraction(real rij_scalar, real rik_scalar,  
                                                      real exp_rij_scalar, real exp_rik_scalar, 
                                                      real3 rji) {
            // rij_scalar denotes the scalar of vector rij
            // rji is the vector from atom j to atom i
            // rik_scalar denotes the scalar of some other vector rik

            // and later we will multiply this by the scalar multiplier associated with the contribution 
            // -- i.e., type A, type B, or type C
            //real forceScalar =  dswitchdr(rij_scalar) * switching(rik_scalar) * exp_rij_scalar * exp_rik_scalar ;
            //forceScalar -= (k3 * switching(rij_scalar) * switching(rik_scalar) * exp_rij_scalar * exp_rik_scalar);
            real forceScalar =  dswitchdr(rij_scalar) ; 
            forceScalar -= (k3 * switching(rij_scalar)) ;
            forceScalar *=  ((switching(rik_scalar) * exp_rij_scalar * exp_rik_scalar) * (1.0 / rij_scalar));
            return forceScalar * rji;
        }
        
        // pass fs_a1_sum by reference b/c we return it to the function modified

        /* old notation (we now use same notation as KS 2008)
        __device__ void threeBodyForce(real3 &fs_a1_sum, real3 &fs_b1_sum, real3 &fs_c1_sum,
                                       real fs_a1b2_scalar, real fs_a1c2_scalar,
                                       real fs_a2b1_scalar, real fs_a2c1_scalar,
                                       real3 r_a1b2, real r_a1b2_scalar,
                                       real3 r_a1c2, real r_a1c2_scalar,
                                       real3 r_a1b3, real r_a1b3_scalar,
                                       real3 r_a1c3, real r_a1c3_scalar,
                                       real3 r_a2b1, real r_a2b1_scalar,
                                       real3 r_a2c1, real r_a2c1_scalar,
                                       real3 r_a2b3, real r_a2b3_scalar,
                                       real3 r_a2c3, real r_a2c3_scalar,
                                       real3 r_a3b1, real r_a3b1_scalar,
                                       real3 r_a3c1, real r_a3c1_scalar, 
                                       real3 r_a3b2, real r_a3b2_scalar, 
                                       real3 r_a3c2, real r_a3c2_scalar) {

        */
        __device__ void threeBodyForce(real3 &fs_a1_sum, real3 &fs_b1_sum, real3 &fs_c1_sum,
                                        real fs_b2a1_scalar, real fs_c2a1_scalar,
                                        real fs_b1a2_scalar, real fs_c1a2_scalar,
                                        real3 r_b2a1, real r_b2a1_scalar,
                                        real3 r_c2a1, real r_c2a1_scalar,
                                        real3 r_b3a1, real r_b3a1_scalar,
                                        real3 r_c3a1, real r_c3a1_scalar,
                                        real3 r_b1a2, real r_b1a2_scalar,
                                        real3 r_c1a2, real r_c1a2_scalar,
                                        real3 r_b3a2, real r_b3a2_scalar,
                                        real3 r_c3a2, real r_c3a2_scalar,
                                        real3 r_b1a3, real r_b1a3_scalar,
                                        real3 r_c1a3, real r_c1a3_scalar, 
                                        real3 r_b2a3, real r_b2a3_scalar, 
                                        real3 r_c2a3, real r_c2a3_scalar) {


            // so, first, we compute the force scalars associated with each of the 12 unique intermolecular OH distances
            // -- note: we already passed in a1b2, a1c2, a2b1, and a2c1 evaluations of the exp(-k3 r) value
            // get the a1b3 and a1c3 terms
            real fs_b3a1_scalar = threeBodyForceScalar(r_b3a1_scalar);
            real fs_c3a1_scalar = threeBodyForceScalar(r_c3a1_scalar);
        
            // get the a2b3 and a2c3 terms
            real fs_b3a2_scalar = threeBodyForceScalar(r_b3a2_scalar);
            real fs_c3a2_scalar = threeBodyForceScalar(r_c3a2_scalar);

            // get the a3b1, a3c1, a3b2, and a3c2 terms
            real fs_b1a3_scalar = threeBodyForceScalar(r_b1a3_scalar);
            real fs_c1a3_scalar = threeBodyForceScalar(r_c1a3_scalar);
            real fs_b2a3_scalar = threeBodyForceScalar(r_b2a3_scalar);
            real fs_c2a3_scalar = threeBodyForceScalar(r_c2a3_scalar);

            /* old notation
            // get the a1b3 and a1c3 terms
            real fs_a1b3_scalar = threeBodyForceScalar(r_a1b3_scalar);
            real fs_a1c3_scalar = threeBodyForceScalar(r_a1c3_scalar);
        
            // get the a2b3 and a2c3 terms
            real fs_a2b3_scalar = threeBodyForceScalar(r_a2b3_scalar);
            real fs_a2c3_scalar = threeBodyForceScalar(r_a2c3_scalar);

            // get the a3b1, a3c1, a3b2, and a3c2 terms
            real fs_a3b1_scalar = threeBodyForceScalar(r_a3b1_scalar);
            real fs_a3c1_scalar = threeBodyForceScalar(r_a3c1_scalar);
            real fs_a3b2_scalar = threeBodyForceScalar(r_a3b2_scalar);
            real fs_a3c2_scalar = threeBodyForceScalar(r_a3c2_scalar);

            */
            // make local force sum vectors for the three atoms from A-type contributions
            real3 local_a1_force_sum = make_real3(0.0, 0.0, 0.0);
            real3 local_b1_force_sum = make_real3(0.0, 0.0, 0.0);
            real3 local_c1_force_sum = make_real3(0.0, 0.0, 0.0);
           
            /*
             * TYPE A contributions
             * -- compute the vector additions to a1, b1, and c1 due to A-type interactions
             *    
             * ____________________________________________________________________________
             * Some notes on notation:
             * - e.g. r_b2a1_scalar = distance between atoms a1 and b2
             *   r_b2a1 = vector from a1 to b2
             *   --- a1 is the reference atom from this perspective
             *   --- since the oxygens are always the reference atom in the distance vectors, and most 
             *       of our force computations are w.r.t. b1 and c1 because there are two of them,
             *       multiply by -1.0 and subsume this term in the force computations
             *
             *  fs_b2a1_scalar is the expf(-k3 * r_b2a1_scalar) that has already been calculated,
             *   and so on for other b*a* terms
             */

            // now, compute the terms unique to a1, the oxygen atom on our reference molecule.
            // ---- first, the A-type anti-cooperative contributions that are unique to a1
            // so, we've already computed the exp(-k3 * rij) for all the 12 O-H distances

            // four interactions of type A involving atom A1
            // ---- NOTE: we pass in the vector (-1.0 * r_b2a1) because we actually want rji rather than rij
            //        --- this way we won't have to put the -1.0 factor for all the interactions for b1 and c1 below 
            real3 local_a1_force_sum_A = threeBodyInteraction(r_b2a1_scalar, r_c2a3_scalar, fs_b2a1_scalar, fs_c2a3_scalar, -1.0 * r_b2a1);
            local_a1_force_sum_A += threeBodyInteraction(r_c2a1_scalar, r_b2a3_scalar, fs_c2a1_scalar, fs_b2a3_scalar, -1.0 * r_c2a1);
            local_a1_force_sum_A += threeBodyInteraction(r_b3a1_scalar, r_c3a2_scalar, fs_b3a1_scalar, fs_c3a2_scalar, -1.0 * r_b3a1);
            local_a1_force_sum_A += threeBodyInteraction(r_c3a1_scalar, r_b3a2_scalar, fs_c3a1_scalar, fs_b3a2_scalar, -1.0 * r_c3a1);
            // when adding the above local sums to the total a1 forces, remember to multiply by prefactor for the A-type interactions!

            // compute the force on b1 due to A-type contributions
            real3 local_b1_force_sum_A = threeBodyInteraction(r_b1a2_scalar, r_c1a3_scalar, fs_b1a2_scalar, fs_c1a3_scalar, r_b1a2);
            local_b1_force_sum_A += threeBodyInteraction(r_b1a3_scalar, r_c1a2_scalar, fs_b1a3_scalar, fs_c1a2_scalar, r_b1a3);

            // compute the force on c1 due to A-type contributions
            real3 local_c1_force_sum_A = threeBodyInteraction(r_c1a3_scalar, r_b1a2_scalar, fs_c1a3_scalar, fs_b1a2_scalar, r_c1a3);
            local_c1_force_sum_A += threeBodyInteraction(r_c1a2_scalar, r_b1a3_scalar, fs_c1a2_scalar, fs_b1a3_scalar, r_c1a2);

            
            /*
             * TYPE B contributions
             * -- compute the vector additions to a1, b1, and c1 due to B-type interactions
             */


            /*
             * TYPE B contributions... for atom a1
             */
            // 16 terms of type B that contribute to the force sum for atom a1
            // -- see KS 2008: 
            // compute force from 
            real3 local_a1_force_sum_B = threeBodyInteraction(r_b3a1_scalar, r_b1a2_scalar, fs_b3a1_scalar, fs_b1a2_scalar, -1.0 * r_b3a1);
            local_a1_force_sum_B += threeBodyInteraction(r_c3a1_scalar, r_b1a2_scalar, fs_c3a1_scalar, fs_b1a2_scalar, -1.0 * r_c3a1);
            local_a1_force_sum_B += threeBodyInteraction(r_b3a1_scalar, r_c1a2_scalar, fs_b3a1_scalar, fs_c1a2_scalar, -1.0 * r_b3a1);
           
            // -- see KS 2008: second row of B terms
            local_a1_force_sum_B += threeBodyInteraction(r_c3a1_scalar, r_c1a2_scalar, fs_c3a1_scalar, fs_c1a2_scalar, -1.0 * r_c3a1);
            local_a1_force_sum_B += threeBodyInteraction(r_b2a1_scalar, r_b1a3_scalar, fs_b2a1_scalar, fs_b1a3_scalar, -1.0 * r_b2a1);
            local_a1_force_sum_B += threeBodyInteraction(r_c2a1_scalar, r_b1a3_scalar, fs_c2a1_scalar, fs_b1a3_scalar, -1.0 * r_c2a1);
           
            // third row of B terms
            local_a1_force_sum_B += threeBodyInteraction(r_b2a1_scalar, r_c1a3_scalar, fs_b2a1_scalar, fs_c1a3_scalar, -1.0 * r_b2a1);
            local_a1_force_sum_B += threeBodyInteraction(r_c2a1_scalar, r_c1a3_scalar, fs_c2a1_scalar, fs_c1a3_scalar, -1.0 * r_c2a1);
            local_a1_force_sum_B += threeBodyInteraction(r_b2a1_scalar, r_b3a2_scalar, fs_b2a1_scalar, fs_b3a2_scalar, -1.0 * r_b2a1);
           
            // fourth row of B terms
            local_a1_force_sum_B += threeBodyInteraction(r_b2a1_scalar, r_c3a2_scalar, fs_b2a1_scalar, fs_c3a2_scalar, -1.0 * r_b2a1);
            local_a1_force_sum_B += threeBodyInteraction(r_c2a1_scalar, r_b3a2_scalar, fs_c2a1_scalar, fs_b3a2_scalar, -1.0 * r_c2a1);
            local_a1_force_sum_B += threeBodyInteraction(r_c2a1_scalar, r_c3a2_scalar, fs_c2a1_scalar, fs_c3a2_scalar, -1.0 * r_c2a1);
           
            // and the last four
            local_a1_force_sum_B += threeBodyInteraction(r_b3a1_scalar, r_b2a3_scalar, fs_b3a1_scalar, fs_b2a3_scalar, -1.0 * r_b3a1);
            local_a1_force_sum_B += threeBodyInteraction(r_b3a1_scalar, r_c2a3_scalar, fs_b3a1_scalar, fs_c2a3_scalar, -1.0 * r_b3a1);
            local_a1_force_sum_B += threeBodyInteraction(r_c3a1_scalar, r_b2a3_scalar, fs_c3a1_scalar, fs_b2a3_scalar, -1.0 * r_c3a1);
            local_a1_force_sum_B += threeBodyInteraction(r_c3a1_scalar, r_c2a3_scalar, fs_c3a1_scalar, fs_b2a3_scalar, -1.0 * r_c3a1);
           
            // -- and, multiply those contributions at the end by the corresponding prefactor!
            //
            // end atom a1 type B contributions
            //
            //


            /*
             * 
             * TYPE B contributions: atom b1
             *
             */

            // there are 8 terms of type B that contribute to the force sum for atom b1
            // -- see KS 2008: row 1 of $\Delta E_123 ^B$ expression (page 8314); first two terms
            real3 local_b1_force_sum_B = threeBodyInteraction(r_b1a2_scalar, r_b3a1_scalar, fs_b1a2_scalar, fs_b3a1_scalar, r_b1a2);
            local_b1_force_sum_B += threeBodyInteraction(r_b1a2_scalar, r_c3a1_scalar, fs_b1a2_scalar, fs_c3a1_scalar, r_b1a2);

            // -- see KS 2008: row 2; second and third terms
            local_b1_force_sum_B += threeBodyInteraction(r_b1a3_scalar, r_b2a1_scalar, fs_b1a3_scalar, fs_b2a1_scalar, r_b1a3);
            local_b1_force_sum_B += threeBodyInteraction(r_b1a3_scalar, r_c2a1_scalar, fs_b1a3_scalar, fs_c2a1_scalar, r_b1a3);

            // row 5 first and third terms of B expression
            local_b1_force_sum_B += threeBodyInteraction(r_b1a2_scalar, r_b2a3_scalar, fs_b1a2_scalar, fs_b2a3_scalar, r_b1a2);
            local_b1_force_sum_B += threeBodyInteraction(r_b1a2_scalar, r_c2a3_scalar, fs_b1a2_scalar, fs_c2a3_scalar, r_b1a2);

            // and the last ones... row 7 term 3, and row 8 term 2 in the paper
            local_b1_force_sum_B += threeBodyInteraction(r_b1a3_scalar, r_b3a2_scalar, fs_b1a3_scalar, fs_b3a2_scalar, r_b1a3);
            local_b1_force_sum_B += threeBodyInteraction(r_b1a3_scalar, r_c3a2_scalar, fs_b1a3_scalar, fs_c3a2_scalar, r_b1a3);

            /*
             *
             * end TYPE B contributions for atom b1
             *
             */




            /*
             *
             * TYPE B contributions for atom c1 begin
             *
             */
            // again, 8 terms of type B that contribute to the force sum for atom c1

            // compute g(r_c1a2, r_b3a1) force contribution
            real3 local_c1_force_sum_B = threeBodyInteraction(r_c1a2_scalar, r_b3a1_scalar, fs_c1a2_scalar, fs_b3a1_scalar, r_c1a2);
            local_c1_force_sum_B += threeBodyInteraction(r_c1a2_scalar, r_c3a1_scalar, fs_c1a2_scalar, fs_c3a1_scalar, r_c1a2);
            
            local_c1_force_sum_B += threeBodyInteraction(r_c1a3_scalar, r_b2a1_scalar, fs_c1a3_scalar, fs_b2a1_scalar, r_c1a3);
            local_c1_force_sum_B += threeBodyInteraction(r_c1a3_scalar, r_c2a1_scalar, fs_c1a3_scalar, fs_c2a1_scalar, r_c1a3);
            
            local_c1_force_sum_B += threeBodyInteraction(r_c1a2_scalar, r_b2a3_scalar, fs_c1a2_scalar, fs_b2a3_scalar, r_c1a2);
            local_c1_force_sum_B += threeBodyInteraction(r_c1a2_scalar, r_c2a3_scalar, fs_c1a2_scalar, fs_c2a3_scalar, r_c1a2);
            
            local_c1_force_sum_B += threeBodyInteraction(r_c1a3_scalar, r_b3a2_scalar, fs_c1a3_scalar, fs_b3a2_scalar, r_c1a3);
            local_c1_force_sum_B += threeBodyInteraction(r_c1a3_scalar, r_c3a2_scalar, fs_c1a3_scalar, fs_c3a2_scalar, r_c1a3);

            /*
             *
             * TYPE B contributions for atom c1 end
             *
             */


            /* TYPE C Contributions
             * -- need to compute the type C anti-cooperative interactions for each of the atoms a1, b1, c1
             */
            
            /*
             * begin TYPE C contributions to atom a1
             */
            // 4 terms contribute to this expression for atom a1; BUT, they have double contribution!
            real3 local_a1_force_sum_C = threeBodyInteraction(r_b2a1_scalar, r_b3a1_scalar, fs_b2a1_scalar, fs_b3a1_scalar, -1.0 * r_b2a1);
            local_a1_force_sum_C += threeBodyInteraction(r_b3a1_scalar, r_b2a1_scalar, fs_b3a1_scalar, fs_b2a1_scalar, -1.0 * r_b3a1);

            local_a1_force_sum_C += threeBodyInteraction(r_b2a1_scalar, r_c3a1_scalar, fs_b2a1_scalar, fs_c3a1_scalar, -1.0 * r_b2a1);
            local_a1_force_sum_C += threeBodyInteraction(r_c3a1_scalar, r_b2a1_scalar, fs_c3a1_scalar, fs_b2a1_scalar, -1.0 * r_c3a1);

            local_a1_force_sum_C += threeBodyInteraction(r_c2a1_scalar, r_b3a1_scalar, fs_c2a1_scalar, fs_b3a1_scalar, -1.0 * r_c2a1);
            local_a1_force_sum_C += threeBodyInteraction(r_b3a1_scalar, r_c2a1_scalar, fs_b3a1_scalar, fs_c2a1_scalar, -1.0 * r_b3a1);


            local_a1_force_sum_C += threeBodyInteraction(r_c2a1_scalar, r_c3a1_scalar, fs_c2a1_scalar, fs_c3a1_scalar, -1.0 * r_c2a1);
            local_a1_force_sum_C += threeBodyInteraction(r_c3a1_scalar, r_c2a1_scalar, fs_c3a1_scalar, fs_c2a1_scalar, -1.0 * r_c3a1);
            // end TYPE C contributions to atom a1



            /*
             * begin TYPE C contributions to atom b1
             */
            // -- four of these contribute to the force on atom b1
            real3 local_b1_force_sum_C = threeBodyInteraction(r_b1a2_scalar, r_b3a2_scalar, fs_b1a2_scalar, fs_b3a2_scalar, r_b1a2);
            local_b1_force_sum_C += threeBodyInteraction(r_b1a2_scalar, r_c3a2_scalar, fs_b1a2_scalar, fs_c3a2_scalar, r_b1a2);

            local_b1_force_sum_C += threeBodyInteraction(r_b1a3_scalar, r_b2a3_scalar, fs_b1a3_scalar, fs_b2a3_scalar, r_b1a3);
            local_b1_force_sum_C += threeBodyInteraction(r_b1a3_scalar, r_c2a3_scalar, fs_b1a3_scalar, fs_c2a3_scalar, r_b1a3);
            // -- end TYPE C contributions to atom b1


            /*
             * begin TYPE C contributions to atom c1
             */ 
            real3 local_c1_force_sum_C = threeBodyInteraction(r_c1a2_scalar, r_b3a2_scalar, fs_c1a2_scalar, fs_b3a2_scalar, r_c1a2);
            local_c1_force_sum_C += threeBodyInteraction(r_c1a2_scalar, r_c3a2_scalar, fs_c1a2_scalar, fs_c3a2_scalar, r_c1a2);

            local_c1_force_sum_C += threeBodyInteraction(r_c1a3_scalar, r_b2a3_scalar, fs_c1a3_scalar, fs_b2a3_scalar, r_c1a3);
            local_c1_force_sum_C += threeBodyInteraction(r_c1a3_scalar, r_c2a3_scalar, fs_c1a3_scalar, fs_c2a3_scalar, r_c1a3);
            // END TYPE C contributions to C1


            /*
             *
             *
             * Add the local contributions to the original fs_a1_sum, fs_b1_sum, and fs_c1_sum after prefactors have been
             * incorporated to the local sums computed above
             *
             *
             */
            local_a1_force_sum += (Ea * local_a1_force_sum_A);
            local_a1_force_sum += (Eb * local_a1_force_sum_B);
            local_a1_force_sum += (Ec * local_a1_force_sum_C);

            local_b1_force_sum += (Ea * local_b1_force_sum_A);
            local_b1_force_sum += (Eb * local_b1_force_sum_B);
            local_b1_force_sum += (Ec * local_b1_force_sum_C);

            local_c1_force_sum += (Ea * local_c1_force_sum_A);
            local_c1_force_sum += (Eb * local_c1_force_sum_B);
            local_c1_force_sum += (Ec * local_c1_force_sum_C);

            // and now add the proper force sums to the original sums that were passed by reference above


            fs_a1_sum += local_a1_force_sum;
            fs_b1_sum += local_b1_force_sum;
            fs_c1_sum += local_c1_force_sum;

        } // end threeBodyForce(*bigArgsListHere)
                                        
};

// need to get a per particle energy kernel as well... copy and paste!



#endif /* EVALUATOR_E3B3 */
