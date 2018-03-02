#pragma once
#ifndef EVALUATOR_LJFS
#define EVALUATOR_LJFS
//Force-shifted Lennard-Jones Pair potential

#include "cutils_math.h"

void export_EvaluatorLJFS();

class EvaluatorLJFS {
    public:
        
        EvaluatorLJFS();

        inline __host__ __device__ real3 force(real3 dr, real params[4], real lenSqr, real multiplier) {
            if (multiplier) {
                real epstimes24 = params[1];
                real sig6 = params[2];
                real p1 = epstimes24*2.0*sig6*sig6;
                real p2 = epstimes24*sig6;
                real r2inv = 1.0/lenSqr;
                real r6inv = r2inv*r2inv*r2inv;
                // force scalar is as LJ - f(rc)
                real forceScalar = (r6inv * r2inv * (p1 * r6inv - p2)-params[3]/sqrt(lenSqr)) * multiplier ;

                return dr * forceScalar;
            }
            return make_real3(0, 0, 0);
        }
        
        inline __host__ __device__ real energy(real params[4], real lenSqr, real multiplier) {
            if (multiplier) {
                // params[1] holds 24.0 * epsilon
                real four_eps = params[1] / 6.0;
                real sig6 = params[2];
                real r2inv = 1/lenSqr;
                real r6inv = r2inv*r2inv*r2inv;
                real sig6r6inv = sig6 * r6inv;
                // for shifting the energy as well
                real rCutSqr = params[0];
                real rCut6 = rCutSqr*rCutSqr*rCutSqr;
                real sig6InvRCut6 = sig6 / rCut6;
                real offsetOver4Eps = sig6InvRCut6*(sig6InvRCut6-1.0);
                //return 0.5 * 4.0*eps*(sig6r6inv*(sig6r6inv-1.0) - offsetOver4Eps) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
                // U_ljfs = U_lj - U_lj(r_c) - (dU/dr)|_(r_c) * (r_ij - r_c)
                // here, the linear term is as params[3] * (sqrt(r_ij*r_ij) - sqrt(rCut * rCut))
                // and the energy shift to zero at the cutoff is as offsetOver4Eps
                real u_lj_over_4_eps = (sig6r6inv * (sig6r6inv - 1.0));
                real u_lj_shift_linear = params[3] * (sqrt(rCutSqr) - sqrt(lenSqr));
                // the aggregate terms; still need to multiply by 0.5 to account for double counting, and then account for the multiplier (e.g. if a CHARMM 1-3 interaction, etc.)
                real u_lj = four_eps * (u_lj_over_4_eps - offsetOver4Eps) +  u_lj_shift_linear;
                return (0.5 * u_lj * multiplier);
            }
            return 0.0;
        }

        // python interface that calls force function
        __host__ Vector forcePy(double sigma, double epsilon, double rcut, Vector dr);

        // python interface that calls the energy function
        __host__ double energyPy(double sigma, double epsilon, double rcut, double distance);

        // python interface that calls the force function, and calculates on the GPU
        // TODO
        __host__ Vector forcePy_device(double sigma, double epsilon, double rcut, Vector dr);

        // python interface that calls the energy function, and calculates on the GPU
        // TODO
        __host__ double energyPy_device(double sigma, double epsilon, double rcut, double distance);

};

#endif
