#pragma once
#ifndef EVALUATOR_E3B
#define EVALUATOR_E3B

#include "cutils_math.h"
#include "helpers.h"
//void export_EvaluatorE3B();

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
        real minus_rfminusrs_cubed_inv; 
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
        real minus_k2; // -1.0 * k2
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
            // precomputing a few values:
            minus_k3 = -1.0 * k3; // eliminates a -1.0 multiplication that happens every time..
            minus_k2 = -1.0 * k2; // eliminates a -1.0 multiplication that happens every time
            real rfminusrs = rf_ - rs_;
            rfminusrs_cubed_inv = 1.0 / (rfminusrs * rfminusrs * rfminusrs);
            minus_rfminusrs_cubed_inv = -1.0 * rfminusrs_cubed_inv; // for switching function evaluation
            rstimes3 = 3.0 * rs_;
        };

        // implements the O-O two-body correction to TIP4P/2005
        template <bool COMP_VIRIALS>
        inline __host__ __device__ void twoBodyForce(const real3 &dr, real3 &fs, Virial &virial) {
            real r = length(dr);
            real forceScalar = k2 * E2 * exp(minus_k2 * r) / r;
            if (COMP_VIRIALS) {
                real3 force = dr * forceScalar;
                fs += force;
                computeVirial(virial, force, dr);
            } else {
                fs += (dr * forceScalar);
            }
        }

        // 0.5 for double counting
        inline __host__ __device__ real twoBodyCorrectionEnergy(real r) {
#ifdef DASH_DOUBLE 
            return (0.5 * E2 * exp(minus_k2  * r));
#else
            return (0.5 * E2 * expf(minus_k2 * r));
#endif /* DASH_DOUBLE */
        }

        // implements one evaluation of the switching function for smooth cutoff of the potential
        // --- the switching function is /inclusive/ of the bounds - i.e., [rs,rf]
        inline __host__ __device__ real switching(const real& r) {
            if (r <= rs) {
                return 1.0;
            } 
            if (r > rf) {
                return 0.0;
            }
            real val = (rf - r);
            val *= val;
            val *= ((rf + (2.0 * r) - rstimes3) * rfminusrs_cubed_inv);
            return val;
        }

        // this is f(r) and so must be accounted for in taking the derivative of the potential
        inline __host__ __device__ real dswitchdr(const real& r) {
            if (r <= rs) {
                return 0.0;
            } else if (r > rf) {
                return 0.0;
            } else {
                real value = 6.0 * (rf - r) * (rs - r);
                value *= rfminusrs_cubed_inv;
                return value;
            }
        }
        
        inline __host__ __device__ real threeBodyPairEnergy(const real& r) {
#ifdef DASH_DOUBLE
            return (switching(r) *  exp(minus_k3 * r));
#else
            return (switching(r) * expf(minus_k3 * r));
#endif
        }

        inline __host__ __device__ real3 threeBodyForce(const real3& dr, const real& r) {
            if (r <= rf) {
#ifdef DASH_DOUBLE
                real es          = exp(minus_k3 * r);
#else
                real es          = expf(minus_k3 * r);
#endif
                real forceScalar = k3 * es / r ;
                if (r > rs) {
                    real scale = switching(r);
                    forceScalar *= scale;
                    forceScalar -= (es * dswitchdr(r) / r);
                }
                return dr * forceScalar;
            }
            return make_real3(0.0, 0.0, 0.0);
        }

        // TODO: Virials! -- will need to modify fn signature to permit passing dr vectors & virials 
        //  for all the atoms.  Can we do an atomicAdd of the virials..???
        template <bool COMPUTE_VIRIALS>
        inline __host__ __device__ void threeBodyPairForce(const real3& fs_b2a1, const real3& fs_c2a1, 
                                                            const real3& fs_b1a2, const real3& fs_c1a2,
                                                            const real4& totalsThisMol,
                                                            const real4& totalsOtherMol, 
                                                            const real4& esThisPair,
                                                            real3& fs_a1, real3& fs_b1, real3& fs_c1,
                                                            real3& fs_a2, real3& fs_b2, real3& fs_c2) {
                

                real3 fs_a1_local, fs_b1_local, fs_c1_local;
                real3 fs_a2_local, fs_b2_local, fs_c2_local;
                real3 fj;
                real fscale;
                
                fs_a1_local = make_real3(0.0,0.0,0.0);
                fs_b1_local = make_real3(0.0,0.0,0.0);
                fs_c1_local = make_real3(0.0,0.0,0.0);
                
                fs_a2_local = make_real3(0.0,0.0,0.0);
                fs_b2_local = make_real3(0.0,0.0,0.0);
                fs_c2_local = make_real3(0.0,0.0,0.0);
                
                // first, type A: - 
                // first loop (j = 1, k = 1)
                fscale = Ea * (totalsOtherMol.y - esThisPair.x);
                fj = fscale * fs_c2a1;
                fs_a1_local -= fj;
                fs_c2_local += fj;
                
                fscale     = Ea * (totalsThisMol.y - esThisPair.z);
                fj = fscale * fs_c1a2;  // XXX there was an error here! maybe fixed.
                fs_a2_local -= fj;
                fs_c1_local += fj;
                
                // second loop (j=2, k=0)
                fscale     = Ea * (totalsOtherMol.z - esThisPair.y);
                fj = fscale * fs_b2a1;
                fs_a1_local -= fj;
                fs_b2_local += fj;

                fscale     = Ea * (totalsThisMol.z - esThisPair.w);
                fj = fscale * fs_b1a2;
                fs_a2_local -= fj;
                fs_b1_local += fj;

                // ok, now, type B:
                fscale = Eb * (totalsOtherMol.x + totalsThisMol.y + totalsThisMol.z - 2.0 * (esThisPair.z + esThisPair.w));

                fj = fscale * fs_b2a1;
                fs_a1_local -= fj;
                fs_b2_local += fj;

                fj = fscale * fs_c2a1;
                fs_a1_local -= fj;
                fs_c2_local += fj;

                fscale = Eb * (totalsThisMol.x + totalsOtherMol.y + totalsOtherMol.z - 2.0 * (esThisPair.x + esThisPair.y));
                fj = fscale * fs_b1a2;
                fs_a2_local -= fj;
                fs_b1_local += fj;

                fj = fscale * fs_c1a2;
                fs_a2_local -= fj;
                fs_c1_local += fj;

                // ok, now type C:
                fscale = Ec * (totalsThisMol.x - esThisPair.y - esThisPair.x);
                
                fj = fscale * fs_b2a1;
                fs_a1_local -= fj;
                fs_b2_local += fj;

                fj = fscale * fs_c2a1;
                fs_a1_local -= fj;
                fs_c2_local += fj;

                fscale = Ec * (totalsOtherMol.x - esThisPair.z - esThisPair.w);

                fj = fscale * fs_b1a2;
                fs_a2_local -= fj;
                fs_b1_local += fj;

                fj = fscale * fs_c1a2;
                fs_a2_local -= fj;
                fs_c1_local += fj;

                // write to the fj's that we passed by reference;
                // --- we do an /assignment/ for molecule 2's atoms. 
                //         - they are written to memory immediately after exiting this function
                //     we do an /aggregation/ for molecule 1's atoms;
                //     molecule 1's atoms are aggregated during the molecule loop,
                //     and possiby across multiple threads (as the reference atom, and also as the 
                //     neighbor atom.)

                fs_a1 += fs_a1_local;
                fs_b1 += fs_b1_local;
                fs_c1 += fs_c1_local;

                fs_a2 = fs_a2_local;
                fs_b2 = fs_b2_local;
                fs_c2 = fs_c2_local;

                return;
        }

        inline __host__ __device__ void threeBodyPairEnergy(const real4& totalsThisMol,
                                                            const real4& totalsOtherMol, 
                                                            const real4& esThisPair,
                                                            real3& myAtomsEnergies,
                                                            real3& otherAtomsEnergies) {

                real fscale;
                real energy;
                // since we distribute the energies per particle...
                real local_a1_sum, local_b1_sum, local_c1_sum;
                real local_a2_sum, local_b2_sum, local_c2_sum;

                local_a1_sum = 0.0;
                local_b1_sum = 0.0;
                local_c1_sum = 0.0;

                local_a2_sum = 0.0;
                local_b2_sum = 0.0;
                local_c2_sum = 0.0;

                // TODO: verify, again, that we are using correct esThisPair accesses!
                
                // note: factor of 1/4 arises bc we partition it equally between two atoms.
                // in the actual decomposition, a factor of 1/2 arises because we assigned total energies duplicitously 
                // in the 'total' arrays (which is required).
                
                // first, type A: - 
                // first loop (j = 1, k = 1)
                fscale = Ea * (totalsOtherMol.y - esThisPair.x);
                energy = 0.25 * fscale * esThisPair.y;  
                local_a1_sum += energy;
                local_c2_sum += energy;
                
                fscale     = Ea * (totalsThisMol.y - esThisPair.z);
                energy = 0.25 * fscale * esThisPair.w;
                local_a2_sum += energy;
                local_c1_sum += energy;

                // second loop (j=2, k=0)
                fscale     = Ea * (totalsOtherMol.z - esThisPair.y);
                energy = 0.25 * fscale * esThisPair.x;
                local_a1_sum += energy;
                local_b2_sum += energy;

                fscale     = Ea * (totalsThisMol.z - esThisPair.w);
                energy = 0.25 * fscale * esThisPair.z;
                local_a2_sum += energy;
                local_b1_sum += energy;

                // ok, now, type B:
                fscale = Eb * (totalsOtherMol.x + totalsThisMol.y + totalsThisMol.z - 2.0 * (esThisPair.z + esThisPair.w));
                energy = 0.25 * fscale * esThisPair.x;
                local_a1_sum += energy;
                local_b2_sum += energy;

                energy = 0.25 * fscale * esThisPair.y;
                local_a1_sum += energy;
                local_c2_sum += energy;

                fscale = Eb * (totalsThisMol.x + totalsOtherMol.y + totalsOtherMol.z - 2.0 * (esThisPair.x + esThisPair.y));
                energy = 0.25 * fscale * esThisPair.z;
                local_a2_sum += energy;
                local_b1_sum += energy;

                energy = 0.25 * fscale * esThisPair.w;
                local_a2_sum += energy;
                local_c1_sum += energy;

                // ok, now type C:
                fscale = Ec * (totalsThisMol.x - esThisPair.y - esThisPair.x);
                energy = 0.25 * fscale * esThisPair.x;
                local_a1_sum += energy;
                local_b2_sum += energy;

                energy = 0.25 * fscale * esThisPair.y;
                local_a1_sum += energy;
                local_c2_sum += energy;

                fscale = Ec * (totalsOtherMol.x - esThisPair.z - esThisPair.w);
                energy = 0.25 * fscale * esThisPair.z;
                local_a2_sum += energy;
                local_b1_sum += energy;

                energy = 0.25 * fscale * esThisPair.w;
                local_a2_sum += energy;
                local_c1_sum += energy;
                
                // do assignment to molecule 2's energies, then they are added atomically to global data
                // -- molecule 1's energies are summed across MULTITHREADPERATOM, if necessary, 
                //    then written to global memory 
                otherAtomsEnergies = make_real3(local_a2_sum, local_b2_sum, local_c2_sum);
                myAtomsEnergies += make_real3(local_a1_sum, local_b1_sum, local_c1_sum);

                return;
        }
};

#endif /* EVALUATOR_E3B */
